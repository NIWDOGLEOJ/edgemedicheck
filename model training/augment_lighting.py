#!/usr/bin/env python3
"""
Offline Data Augmentation Script for Counterfeit Pharmacy Images.

Generates realistic lighting, brightness, shadow, and color temperature variations 
for training images to expand sample size and increase model robustness against real-world lighting conditions.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

BASE_DIR = Path(__file__).parent / "dataset" / "train"
CLASSES = ["Fake", "Real"]

def apply_lighting_transformations(image_path, output_dir):
    """
    Generate multiple lighting variants for a given image file:
    1. Bright Lighting (Over-exposure / Direct spotlight)
    2. Dim Lighting (Low-light environment / Shadow)
    3. Warm Lighting (Tungsten / Warm indoor lamp tint)
    4. Cool Lighting (Fluorescent / Daylight LED tint)
    5. Contrast Boost & Vignette (Uneven lighting distribution)
    """
    try:
        # Load image with PIL & OpenCV
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        stem = image_path.stem
        ext = image_path.suffix if image_path.suffix.lower() in {".jpg", ".png", ".jpeg"} else ".png"

        # 1. Bright Lighting (1.4x brightness)
        enhancer = ImageEnhance.Brightness(img_pil)
        img_bright = enhancer.enhance(1.4)
        img_bright.save(output_dir / f"{stem}_aug_bright{ext}")

        # 2. Dim / Low Lighting (0.6x brightness)
        img_dim = enhancer.enhance(0.6)
        img_dim.save(output_dir / f"{stem}_aug_dim{ext}")

        # 3. Warm Lighting (Shift RGB channels towards yellow/red)
        warm_np = img_np.astype(np.float32)
        warm_np[:, :, 0] = np.clip(warm_np[:, :, 0] * 1.15, 0, 255) # Red booster
        warm_np[:, :, 1] = np.clip(warm_np[:, :, 1] * 1.05, 0, 255) # Green slight boost
        warm_np[:, :, 2] = np.clip(warm_np[:, :, 2] * 0.85, 0, 255) # Blue drop
        Image.fromarray(warm_np.astype(np.uint8)).save(output_dir / f"{stem}_aug_warm{ext}")

        # 4. Cool Lighting (Shift RGB channels towards blue)
        cool_np = img_np.astype(np.float32)
        cool_np[:, :, 0] = np.clip(cool_np[:, :, 0] * 0.85, 0, 255) # Red drop
        cool_np[:, :, 1] = np.clip(cool_np[:, :, 1] * 1.05, 0, 255) # Green
        cool_np[:, :, 2] = np.clip(cool_np[:, :, 2] * 1.20, 0, 255) # Blue booster
        Image.fromarray(cool_np.astype(np.uint8)).save(output_dir / f"{stem}_aug_cool{ext}")

        # 5. Shadow Gradient / Directional Lighting (Overhead shadow gradient)
        h, w, _ = img_np.shape
        gradient = np.linspace(0.6, 1.2, h).reshape(h, 1, 1)
        grad_np = np.clip(img_np.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        Image.fromarray(grad_np).save(output_dir / f"{stem}_aug_gradient{ext}")

        return 5
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def run_augmentation():
    print("==================================================")
    print("Generating Lighting Augmentations for Training Set")
    print("==================================================")

    total_created = 0

    for cls in CLASSES:
        cls_dir = BASE_DIR / cls
        if not cls_dir.exists():
            continue

        # Get original non-augmented images
        original_files = [p for p in cls_dir.glob("*") if p.is_file() and not p.name.startswith(".") and not "_aug_" in p.name]
        print(f"\nProcessing class [{cls}]: {len(original_files)} original images...")

        cls_created = 0
        for img_path in original_files:
            cls_created += apply_lighting_transformations(img_path, cls_dir)

        total_created += cls_created
        final_count = len([p for p in cls_dir.glob("*") if p.is_file() and not p.name.startswith(".")])
        print(f"  -> Generated {cls_created} new lighting variant images.")
        print(f"  -> Total [{cls}] training samples now: {final_count}")

    print("\n--------------------------------------------------")
    print(f"SUCCESS: Created {total_created} augmented lighting images!")
    print(f"Total training dataset size is now expanded!")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_augmentation()
