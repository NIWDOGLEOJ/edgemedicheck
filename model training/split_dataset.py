#!/usr/bin/env python3
"""
Dataset Split & Organization Script for Counterfeit Pharmacy Detection.

This script:
1. Identifies all unique images from dataset/Fake and dataset/Real (ignoring non-image files).
2. Performs a reproducible 70% Train / 15% Val / 15% Test stratified split.
3. Cleans and populates dataset/train, dataset/val, and dataset/test without any data leakage.
4. Validates zero cross-set leakage and outputs exact statistics.
"""

import os
import shutil
import hashlib
import random
from pathlib import Path

# Target ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

BASE_DIR = Path(__file__).parent / "dataset"
CLASSES = ["Fake", "Real"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def get_image_files(class_name):
    """Collect unique images for a given class across top-level and split subfolders."""
    search_dirs = [
        BASE_DIR / class_name,
        BASE_DIR / "train" / class_name,
        BASE_DIR / "val" / class_name,
        BASE_DIR / "test" / class_name
    ]
    
    unique_files = {} # hash -> Path
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for file_path in s_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                if file_path.name.startswith("."):
                    continue
                # Compute hash to avoid duplicating identical files under different names
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash not in unique_files:
                    unique_files[file_hash] = file_path
                    
    return list(unique_files.values())

def split_and_organize():
    random.seed(SEED)
    print(f"Starting dataset reorganization with seed={SEED}...")
    print(f"Split Ratios -> Train: {TRAIN_RATIO*100:.0f}%, Val: {VAL_RATIO*100:.0f}%, Test: {TEST_RATIO*100:.0f}%\n")
    
    stats = {}
    
    for cls in CLASSES:
        images = get_image_files(cls)
        # Shuffle deterministically
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        stats[cls] = {
            "Total": n_total,
            "Train": len(train_imgs),
            "Val": len(val_imgs),
            "Test": len(test_imgs),
            "Train_files": train_imgs,
            "Val_files": val_imgs,
            "Test_files": test_imgs
        }
        
        print(f"[{cls}] Found {n_total} unique images.")
        print(f"  -> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    # Clear existing train, val, test subdirectories
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            target_dir = BASE_DIR / split / cls
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            
    # Copy images to respective target folders
    print("\nCopying files into clean train / val / test structure...")
    for cls in CLASSES:
        for split in ["Train", "Val", "Test"]:
            target_dir = BASE_DIR / split.lower() / cls
            for src_path in stats[cls][f"{split}_files"]:
                dst_path = target_dir / src_path.name
                # Avoid collision if two files have same name but different content
                counter = 1
                while dst_path.exists():
                    dst_path = target_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
                    counter += 1
                shutil.copy2(src_path, dst_path)

    # Verification Step: Check for Leakage
    print("\n--- Verifying Dataset Integrity ---")
    train_hashes, val_hashes, test_hashes = set(), set(), set()
    
    for split, h_set in zip(["train", "val", "test"], [train_hashes, val_hashes, test_hashes]):
        for cls in CLASSES:
            for p in (BASE_DIR / split / cls).glob("*"):
                if p.is_file() and not p.name.startswith("."):
                    with open(p, "rb") as f:
                        h_set.add(hashlib.md5(f.read()).hexdigest())

    leakage_train_val = len(train_hashes.intersection(val_hashes))
    leakage_train_test = len(train_hashes.intersection(test_hashes))
    leakage_val_test = len(val_hashes.intersection(test_hashes))

    print(f"Train vs Val Data Leakage: {leakage_train_val} overlap")
    print(f"Train vs Test Data Leakage: {leakage_train_test} overlap")
    print(f"Val vs Test Data Leakage: {leakage_val_test} overlap")

    if leakage_train_val == 0 and leakage_train_test == 0 and leakage_val_test == 0:
        print("SUCCESS: Data split is completely clean with 0% leakage!")
    else:
        print("WARNING: Data leakage detected!")

    print("\n--- FINAL DATASET SUMMARY ---")
    print(f"{'Class':<10} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 50)
    for cls in CLASSES:
        tr = stats[cls]['Train']
        va = stats[cls]['Val']
        te = stats[cls]['Test']
        tot = stats[cls]['Total']
        print(f"{cls:<10} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")
    print("-" * 50)
    total_tr = sum(stats[c]['Train'] for c in CLASSES)
    total_va = sum(stats[c]['Val'] for c in CLASSES)
    total_te = sum(stats[c]['Test'] for c in CLASSES)
    total_all = sum(stats[c]['Total'] for c in CLASSES)
    print(f"{'Total':<10} | {total_tr:<8} | {total_va:<8} | {total_te:<8} | {total_all:<8}")

if __name__ == "__main__":
    split_and_organize()
