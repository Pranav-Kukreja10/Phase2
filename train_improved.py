"""
Improved Underwater Semantic Segmentation - Phase 2 Task B
- SMP Unet (ResNet34 pretrained encoder)
- Class-weighted Dice + Focal Loss
- Strong underwater augmentations
- AMP + gradient accumulation
- LR scheduler + early stopping
- Robust data loading with fallback matching
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
import re
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

# ================= CONFIG =================
class Config:
    IMAGE_DIR = '/nlsasfs/home/gpucbh/vyakti24/images'
    MASK_DIR  = '/nlsasfs/home/gpucbh/vyakti24/masks/combined'
    NUM_CLASSES = 8
    INPUT_SIZE = (320, 256)  # (H, W)
    BATCH_SIZE = 6  # Increased from 4
    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    VAL_SPLIT = 0.15
    SAVE_DIR = "./checkpoints_improved"
    BEST_MODEL = "best_model_improved.pth"
    ACCUM_STEPS = 2
    AMP = True
    SEED = 42

cfg = Config()

# ================= DEVICE =================
def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    print(f"Using device: {device}")
    return device

device = setup()

# ================= ATOMIC SAVE =================
def atomic_save(obj, path):
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[WARN] Save failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)

# ================= DATASET =================
class UnderwaterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, files, train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        base = os.path.splitext(name)[0]

        # Load image
        try:
            img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        except Exception:
            img = Image.new("RGB", cfg.INPUT_SIZE[::-1], (0,0,0))

        # Find and load mask with fallback matching
        mask_path = None
        for e in ['.png', '.bmp', '.jpg']:
            p = os.path.join(self.mask_dir, base + e)
            if os.path.exists(p):
                mask_path = p
                break

        # Fallback: numeric suffix matching
        if mask_path is None:
            m = re.search(r"(\d{3,7})", base)
            if m:
                num = m.group(1)
                for fname in os.listdir(self.mask_dir):
                    if num in fname:
                        mask_path = os.path.join(self.mask_dir, fname)
                        break

        try:
            if mask_path:
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", img.size, 0)
        except Exception:
            mask = Image.new("L", img.size, 0)

        # Resize
        img = img.resize(cfg.INPUT_SIZE[::-1], Image.BILINEAR)
        mask = mask.resize(cfg.INPUT_SIZE[::-1], Image.NEAREST)

        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.clip(np.array(mask), 0, cfg.NUM_CLASSES - 1).astype(np.uint8)

        # Strong underwater augmentations
        if self.train:
            # Flip
            if random.random() < 0.5:
                img_np = np.fliplr(img_np).copy()
                mask_np = np.fliplr(mask_np).copy()

            # Brightness/contrast
            if random.random() < 0.4:
                alpha = random.uniform(0.7, 1.3)
                img_np = np.clip(img_np.astype(float) * alpha, 0, 255).astype(np.uint8)

            # Gamma
            if random.random() < 0.3:
                gamma = random.uniform(0.8, 1.4)
                img_np = np.clip(255.0 * (img_np / 255.0) ** gamma, 0, 255).astype(np.uint8)

            # Gaussian blur
            if random.random() < 0.2:
                from scipy import ndimage
                img_np = ndimage.gaussian_filter(img_np, sigma=random.uniform(0.5, 1.5))
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # Color jitter
            if random.random() < 0.3:
                for c in range(3):
                    img_np[:,:,c] = np.clip(img_np[:,:,c].astype(float) * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)

        # Convert to tensor
        img_tensor = transforms.ToTensor()(img_np)
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor

# ================= LOSS =================
class WeightedDiceFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='mean')
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, p, t):
        # CE loss
        ce = self.ce(p, t)
        
        # Focal term
        pt = F.softmax(p, dim=1)
        t1 = F.one_hot(t, cfg.NUM_CLASSES).permute(0,3,1,2).float()
        pt_t = (pt * t1).sum(1)
        focal = (self.alpha * (1 - pt_t) ** self.gamma).mean()

        # Dice loss
        p_soft = F.softmax(p, dim=1)
        inter = (p_soft * t1).sum((2,3))
        union = p_soft.sum((2,3)) + t1.sum((2,3))
        dice = 1 - ((2*inter + 1e-6) / (union + 1e-6))

        return ce + 0.5*dice.mean() + 0.5*focal

# ================= METRICS =================
def metrics(pred, gt):
    """Compute mIoU and mF1"""
    ious, f1s = [], []
    p, g = pred.cpu().numpy(), gt.cpu().numpy()
    for c in range(cfg.NUM_CLASSES):
        pc, gc = p==c, g==c
        inter = (pc & gc).sum()
        union = (pc | gc).sum()
        ious.append(inter/union if union>0 else np.nan)
        tp = inter
        fp = (pc & ~gc).sum()
        fn = (~pc & gc).sum()
        f1 = (2*tp) / (2*tp + fp + fn + 1e-6) if (tp + fp + fn) > 0 else np.nan
        f1s.append(f1)
    return np.nanmean(ious), np.nanmean(f1s)

# ================= TRAIN =================
def train():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # Load files with path fallback
    if not os.path.exists(cfg.IMAGE_DIR) or not os.path.exists(cfg.MASK_DIR):
        print("[ERROR] Dataset paths not found!")
        return

    files = sorted([f for f in os.listdir(cfg.IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.bmp'))])
    print(f"[INFO] Found {len(files)} training images")

    tr, va = train_test_split(files, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED)

    train_loader = DataLoader(
        UnderwaterDataset(cfg.IMAGE_DIR, cfg.MASK_DIR, tr, train=True),
        batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        UnderwaterDataset(cfg.IMAGE_DIR, cfg.MASK_DIR, va, train=False),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Build model
    print("[INFO] Building SMP Unet with ResNet34 encoder...")
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=cfg.NUM_CLASSES
    )
    model = model.to(device)

    # Compute class weights
    print("[INFO] Computing class weights...")
    counts = np.zeros(cfg.NUM_CLASSES, dtype=np.float64)
    for fname in tr:
        base = os.path.splitext(fname)[0]
        found = None
        for e in ['.png', '.bmp', '.jpg']:
            p = os.path.join(cfg.MASK_DIR, base + e)
            if os.path.exists(p):
                found = p
                break
        if found is None:
            m = re.search(r"(\d{3,7})", base)
            if m:
                num = m.group(1)
                for f in os.listdir(cfg.MASK_DIR):
                    if num in f:
                        found = os.path.join(cfg.MASK_DIR, f)
                        break
        if found:
            try:
                mask = np.array(Image.open(found).convert('L'))
                for c in range(cfg.NUM_CLASSES):
                    counts[c] += (mask == c).sum()
            except:
                pass

    freq = counts + 1e-6
    weights = freq.sum() / freq
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[INFO] Class weights: {weights}")

    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = WeightedDiceFocalLoss(weight=class_weights, gamma=2.0)
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    
    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.AMP and device.type=='cuda'))

    best_miou = 0
    patience_counter = 0
    max_patience = 15

    for ep in range(cfg.EPOCHS):
        model.train()
        opt.zero_grad()
        
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg.EPOCHS}")):
            x, y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=(cfg.AMP and device.type=='cuda')):
                out = model(x)
                loss = loss_fn(out, y) / cfg.ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

        # Validation
        model.eval()
        miou_total, f1_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                miou, f1 = metrics(out.argmax(1), y)
                miou_total += miou
                f1_total += f1
        
        miou_avg = miou_total / len(val_loader)
        f1_avg = f1_total / len(val_loader)
        
        print(f"Epoch {ep+1} | Val mIoU: {miou_avg:.4f} | Val F1: {f1_avg:.4f} | LR: {opt.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # Early stopping + best model save
        if miou_avg > best_miou:
            best_miou = miou_avg
            patience_counter = 0
            atomic_save(model.state_dict(), os.path.join(cfg.SAVE_DIR, cfg.BEST_MODEL))
            print(f"✓ Best model saved (mIoU: {miou_avg:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"[INFO] Early stopping at epoch {ep+1}")
                break

    # Save final model
    atomic_save(model.state_dict(), os.path.join(cfg.SAVE_DIR, "final_model.pth"))
    print(f"\n[SUCCESS] Training complete! Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    train()
