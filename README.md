# Phase2
What does this project do?
Underwater images are often messy—they are too blue, too dark, or very "foggy". This project uses an AI called a U-Net to:

Fix the colors: Bring back the reds and yellows that the water hides.

Label objects: Automatically draw an outline (segmentation) around things like Fish, Divers, and Reefs.

🧠 How the AI works (Simplified)
The model uses an Encoder-Decoder process, which is like "learning by squinting":

1. The Encoder (Squinting)
The model takes the big image and shrinks it down. By making the image smaller, the AI stops looking at tiny blue dots and starts recognizing the shape of a fish or a diver. It asks: "What am I looking at?".

2. The Decoder (Painting)
The AI then takes those shapes and expands them back to the original size. It uses Skip Connections (shortcuts) to remember exactly where the sharp edges were so the final "label" isn't blurry. It asks: "Where exactly is this object?".

🛠️ How to use the code
Step 1: Training (Teaching the AI)
The AI looks at thousands of pairs of images: a "messy" underwater photo and a "perfectly labeled" mask.

It guesses what the label is.

It compares its guess to the real answer.

It fixes its mistakes and tries again (this is called an Epoch).

To run: python train.py

Step 2: Inference (Using the AI)
Once the AI is "smart," you give it new test images it has never seen before. It will automatically create grayscale masks (labels) for you.

Fish might be labeled as the number 0.

Divers might be labeled as the number 4.

To run: python inference.py

🚀 Why this code is "Improved"
It handles bad lighting: It uses special math (Focal Loss) to make sure it doesn't ignore small objects in the dark.

It’s faster: It uses "AMP" to train quickly without needing an expensive super-computer.

It’s smart: It uses a pretrained "ResNet34" brain, meaning it already knows what basic shapes and colors look like before it even starts underwater training.
