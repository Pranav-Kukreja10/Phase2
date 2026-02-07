"""
Segmentation Inference Script - Submission Format
Runs trained segmentation model (SMP Unet or AttentionUNet) on test images
Outputs .bmp masks matching test image names exactly (required for evaluation)

Output format:
  - .bmp files only
  - Single-channel grayscale (8-bit per pixel)
  - Pixel values = class indices (0-7)
  - Filenames match test image names exactly
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import re

# ================= CONFIG =================
TEST_INPUT = "./underwater_test_data"      # Test images directory
OUTPUT_MASKS = "./submission_masks"       # Where to save predicted masks (.bmp)
MODEL_PATH = "best_model_improved.pth"  # Trained model checkpoint
IMG_SIZE = (320, 256)  # Must match training size (H, W)
NUM_CLASSES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class legend
CLASS_NAMES = [
    "Fish",
    "Reefs",
    "Plants",
    "Ruins",
    "Divers",
    "Robots",
    "Sea-floor",
    "Background"
]

# ================= LOAD MODEL =================
def load_segmentation_model():
    """Auto-detect model architecture from checkpoint and load"""
    
    # Load checkpoint
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Checkpoint not found at {MODEL_PATH}")
        return None

    print(f"[INFO] Loading checkpoint from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Detect which architecture the checkpoint belongs to
    is_attention_unet = any(k.startswith(('e1.', 'e2.', 'e3.', 'e4.', 'att', 'aspp')) for k in state_dict.keys())
    is_smp_unet = any(k.startswith(('encoder.', 'decoder.')) for k in state_dict.keys())
    
    # Load the correct architecture
    try:
        if is_attention_unet:
            print("[INFO] Detected AttentionUNet architecture")
            from train_enhanced import AttentionUNet
            model = AttentionUNet()
        elif is_smp_unet:
            print("[INFO] Detected SMP Unet (resnet34) architecture")
            import segmentation_models_pytorch as smp
            model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
        else:
            print("[WARN] Could not detect architecture, trying AttentionUNet...")
            from train_enhanced import AttentionUNet
            model = AttentionUNet()
        
        model = model.to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print("[INFO] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return None

# ================= INFERENCE =================
def preprocess_image(img_path):
    """Load and preprocess image for model input"""
    try:
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)  # (W,H)
        img_np = np.array(img_resized, dtype=np.uint8)
        
        # Normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img_np).unsqueeze(0).to(DEVICE)
        return img_tensor, (img.size[1], img.size[0])  # (orig_H, orig_W)
    except Exception as e:
        print(f"[ERROR] Could not load {img_path}: {e}")
        return None, None

def infer_segmentation():
    """Run inference on test dataset and output .bmp masks matching test image names"""
    
    # Create output dir
    os.makedirs(OUTPUT_MASKS, exist_ok=True)
    
    # Load model
    model = load_segmentation_model()
    if model is None:
        return
    
    # Get test images
    if not os.path.exists(TEST_INPUT):
        print(f"[ERROR] Test input directory not found: {TEST_INPUT}")
        return
    
    img_files = sorted([f for f in os.listdir(TEST_INPUT) if f.lower().endswith(('.jpg', '.png', '.bmp'))])
    if len(img_files) == 0:
        print(f"[ERROR] No images found in {TEST_INPUT}")
        return
    
    print(f"[INFO] Found {len(img_files)} test images")
    
    # Run inference
    with torch.no_grad():
        for img_name in tqdm(img_files, desc="Inferencing"):
            img_path = os.path.join(TEST_INPUT, img_name)
            
            # Preprocess
            input_tensor, orig_shape = preprocess_image(img_path)
            if input_tensor is None:
                continue
            orig_h, orig_w = orig_shape
            
            # Forward pass
            pred = model(input_tensor)  # (1, NUM_CLASSES, H, W)
            pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()  # (H, W)
            
            # Resize back to original size
            pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Save mask as .bmp with exact image name (required for submission)
            # Replace image extension with .bmp
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(OUTPUT_MASKS, f"{base_name}.bmp")
            
            # Ensure single-channel grayscale (class indices 0-7)
            cv2.imwrite(mask_path, pred_mask_resized)
    
    print(f"\n[SUCCESS] Segmentation masks (.bmp) saved to: {OUTPUT_MASKS}")
    print(f"[INFO] Submission format:")
    print(f"  - Format: Single-channel .bmp files")
    print(f"  - Naming: Exact match with test image names (e.g., img_0000.bmp)")
    print(f"  - Values: Class indices 0-{NUM_CLASSES-1} ({', '.join(CLASS_NAMES)})") 
    print(f"[INFO] Files are ready for submission")

if __name__ == "__main__":
    infer_segmentation()
