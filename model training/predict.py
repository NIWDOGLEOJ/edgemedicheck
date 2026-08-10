#!/usr/bin/env python3
"""
Inference Script for Counterfeit Pharmacy Detection.

Usage:
    ./venv/bin/python3 predict.py --image path/to/medicine.jpg
    ./venv/bin/python3 predict.py --dir dataset/test/Fake
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

OUTPUT_DIR = Path(__file__).parent / "output"
MODEL_PATH = OUTPUT_DIR / "best_counterfeit_model.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model():
    if not MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please run train_model.py first!")
        sys.exit(1)
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint.get("class_names", ["Fake", "Real"])
    
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(class_names))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    return model, class_names

def predict_single_image(model, class_names, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return
        
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        
    pred_label = class_names[pred_idx.item()]
    conf_pct = confidence.item() * 100
    
    print(f"\nImage: {image_path}")
    print(f"  Prediction:  {pred_label.upper()}")
    print(f"  Confidence:  {conf_pct:.2f}%")
    print(f"  Breakdown:   Fake: {probabilities[0]*100:.2f}% | Real: {probabilities[1]*100:.2f}%")
    return pred_label, conf_pct

def main():
    parser = argparse.ArgumentParser(description="Predict Counterfeit vs Real Pharmacy Images")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--dir", type=str, help="Path to a directory containing images")
    args = parser.parse_args()

    if not args.image and not args.dir:
        # Default test sample
        sample_fake = Path(__file__).parent / "dataset" / "test" / "Fake"
        fake_files = list(sample_fake.glob("*.png")) + list(sample_fake.glob("*.jpg"))
        if fake_files:
            args.image = str(fake_files[0])
        else:
            parser.print_help()
            sys.exit(1)

    model, class_names = load_trained_model()

    if args.image:
        predict_single_image(model, class_names, Path(args.image))
    elif args.dir:
        dir_path = Path(args.dir)
        files = [p for p in dir_path.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        print(f"Found {len(files)} images in {dir_path}")
        for p in files[:10]: # Predict first 10
            predict_single_image(model, class_names, p)

if __name__ == "__main__":
    main()
