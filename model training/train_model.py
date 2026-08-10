#!/usr/bin/env python3
"""
Counterfeit Pharmacy Detection Model Trainer.

Uses Transfer Learning (ResNet-18) with PyTorch on Apple Silicon GPU (MPS) / CPU.
Includes data augmentations, early stopping, test set evaluation, confusion matrix generation,
and comprehensive metrics logging.
"""

import os
import sys
import json
import time
import ssl
from pathlib import Path

# Disable SSL verification for model weight downloads on macOS
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Environment Configuration
# Both are rebound by the CLI; the defaults preserve the original behaviour.
BASE_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "output"
PROBE_DIR = None          # optional out-of-distribution probe set
OUTPUT_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders():
    """Build train, val, and test DataLoaders with appropriate augmentations."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=BASE_DIR / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=BASE_DIR / "val", transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=BASE_DIR / "test", transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_names = train_dataset.classes # e.g. ['Fake', 'Real']
    print(f"Loaded Classes: {class_names}")
    print(f"Sample Counts -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_names

def build_model(num_classes):
    """Load pretrained ResNet-18 and replace classifier head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    # Custom classifier head with Dropout for regularization
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def evaluate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def evaluate_probe(model, class_names):
    """Score the model on real counterfeits it was never trained to expect.

    The training set is built from surrogates we generated ourselves, so a
    high test score is consistent with the model having learned our surrogate
    code rather than counterfeiting. The WHO probe is the check: photographs
    of products confirmed falsified by their manufacturers, taken by
    regulators, unrelated to our pipeline.

    Every probe image is falsified, so the only quantity available here is
    recall. On its own it is meaningless -- a model that always answers
    "Fake" scores 100%. It has to be read next to the false-alarm rate on
    genuine packs from the test set above.
    """
    if PROBE_DIR is None:
        return None
    probe_root = Path(PROBE_DIR)
    if not probe_root.is_dir():
        print(f"\n  (probe directory {probe_root} not found; skipping)")
        return None

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(root=probe_root, transform=tf)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    fake_i = class_names.index("Fake") if "Fake" in class_names else 0
    model.eval()
    flagged, total, confid = 0, 0, []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            flagged += int((preds == fake_i).sum().item())
            confid.extend(probs[:, fake_i].cpu().numpy().tolist())
            total += inputs.size(0)

    rec = flagged / total
    print("\n==========================================")
    print("Transfer to real counterfeits (WHO probe)")
    print("==========================================")
    print(f"  {total} photographs of confirmed falsified products")
    print(f"  flagged as Fake: {flagged}/{total}  = {rec * 100:.1f}%")
    print(f"  mean P(Fake):    {float(np.mean(confid)):.3f}")
    print("\n  Single-class set: this is recall only. Compare it against the")
    print("  false-alarm rate on genuine packs printed above -- a model that")
    print("  simply answers 'Fake' more often moves both numbers together.")
    return {
        "n": total,
        "flagged_fake": flagged,
        "recall_on_falsified": float(rec),
        "mean_p_fake": float(np.mean(confid)),
        "source": "WHO Medical Product Alerts (falsified only)",
    }


def run_dataset_audit(strict=True):
    """Check the dataset for shortcuts before spending time training.

    A binary image classifier reaches very high accuracy by exploiting
    whatever separates the two folders most easily. If that happens to be the
    file format or the average brightness rather than the packaging, the
    training log looks excellent and the model detects nothing. This gate runs
    audit_dataset.py first and refuses to train on a dataset whose classes are
    trivially separable.

    Set strict=False (or pass --force) to train anyway, but treat the resulting
    accuracy as a pipeline check, not as a result.
    """
    import subprocess

    audit = Path(__file__).parent / "audit_dataset.py"
    if not audit.exists():
        print("  (audit_dataset.py not found; skipping the dataset check)")
        return True

    print("Auditing the dataset before training...\n")
    proc = subprocess.run(
        [sys.executable, str(audit), str(BASE_DIR)],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr)

    if proc.returncode == 0:
        return True

    print("=" * 70)
    print("  The audit found problems that make accuracy meaningless.")
    print("=" * 70)
    if strict:
        print("\n  Refusing to train. Fix the dataset, or run with --force to")
        print("  proceed anyway as a pipeline smoke test whose numbers must")
        print("  NOT be reported.\n")
        return False

    print("\n  --force given: training anyway.")
    print("  THE RESULTING ACCURACY IS NOT A VALID RESULT. It measures the")
    print("  shortcut the audit identified, not counterfeit detection.\n")
    return True


def run_training(force=False):
    print(f"\n==========================================")
    print(f"Training Counterfeit Pharmacy Detection Model")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE} | Learning Rate: {LEARNING_RATE}")
    print(f"==========================================\n")

    if not run_dataset_audit(strict=not force):
        return

    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    model = build_model(num_classes=len(class_names))

    # Compute class weights to handle slight imbalance (240 Fake vs 406 Real)
    class_counts = [len(list((BASE_DIR / "train" / c).glob("*"))) for c in class_names]
    total_train = sum(class_counts)
    weights = [total_train / (len(class_names) * c) for c in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Class Weights (Handling Imbalance): {dict(zip(class_names, weights))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_path = OUTPUT_DIR / "best_counterfeit_model.pth"

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% "
              f"|| Val Loss: {va_loss:.4f} | Val Acc: {va_acc*100:.2f}%")

        # Save Best Model Checkpoint
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": class_names,
                "val_acc": va_acc,
                "val_loss": va_loss
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path.name} (Val Loss: {va_loss:.4f})")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds.")

    # Plot Training & Validation Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), history["train_loss"], label="Train Loss", marker="o")
    plt.plot(range(1, NUM_EPOCHS + 1), history["val_loss"], label="Val Loss", marker="s")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o")
    plt.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in history["val_acc"]], label="Val Acc", marker="s")
    plt.title("Accuracy Curves (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300)
    plt.close()
    print(f"Saved training curves plot to {OUTPUT_DIR / 'training_curves.png'}")

    # Evaluate Best Model on HELD-OUT TEST DATASET
    print("\n==========================================")
    print("Evaluating Best Model on Hold-out Test Set")
    print("==========================================")
    
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy()) # probability of positive class 'Real'

    # Compute Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    acc = (sum(1 for t, p in zip(y_true, y_pred) if t == p)) / len(y_true)
    auc = roc_auc_score(y_true, y_probs)

    # ImageFolder sorts alphabetically, so index 0 is 'Fake' and 1 is 'Real',
    # which makes the binary precision/recall above describe the *Real* class.
    # For a counterfeit scanner the operative number is the opposite one: how
    # many counterfeits were caught. Report both, explicitly named.
    fake_i = class_names.index("Fake") if "Fake" in class_names else 0
    real_i = 1 - fake_i
    n_fake = sum(1 for t in y_true if t == fake_i)
    n_real = sum(1 for t in y_true if t == real_i)
    caught = sum(1 for t, p in zip(y_true, y_pred) if t == fake_i and p == fake_i)
    false_alarms = sum(1 for t, p in zip(y_true, y_pred) if t == real_i and p == fake_i)
    fake_recall = caught / n_fake if n_fake else float("nan")
    fpr_genuine = false_alarms / n_real if n_real else float("nan")

    print(f"\nTEST SET RESULTS:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%   (positive class = "
          f"{class_names[real_i]})")
    print(f"Recall:    {recall * 100:.2f}%   (positive class = "
          f"{class_names[real_i]})")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"\n  Counterfeits caught (recall on Fake): "
          f"{fake_recall * 100:.2f}%  ({caught}/{n_fake})")
    print(f"  False alarms on genuine packs:       "
          f"{fpr_genuine * 100:.2f}%  ({false_alarms}/{n_real})")

    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Test Set Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {OUTPUT_DIR / 'confusion_matrix.png'}")

    # Transfer to real counterfeits
    probe = evaluate_probe(model, class_names)

    # Save Metrics JSON
    results = {
        "test_accuracy": float(acc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1_score": float(f1),
        "test_roc_auc": float(auc),
        "test_fake_recall": float(fake_recall),
        "test_false_positive_rate_genuine": float(fpr_genuine),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "ood_probe": probe,
    }
    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved numerical metrics to {OUTPUT_DIR / 'evaluation_results.json'}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the counterfeit classifier.")
    ap.add_argument(
        "--force", action="store_true",
        help="train even when the dataset audit fails. The result is then a "
             "pipeline smoke test, not a reportable number.",
    )
    ap.add_argument(
        "--skip-audit", action="store_true",
        help="do not run the dataset audit at all (not recommended)",
    )
    ap.add_argument(
        "--data", default=None,
        help="dataset root containing train/ val/ test/ (default: ./dataset)",
    )
    ap.add_argument(
        "--probe", default=None,
        help="out-of-distribution probe directory, e.g. probe/who. Every "
             "image in it is a real falsified product, so it yields recall "
             "only; read it beside the false-alarm rate on genuine packs.",
    )
    ap.add_argument(
        "--out", default=None, help="output directory (default: ./output)",
    )
    ap.add_argument("--epochs", type=int, default=None)
    cli = ap.parse_args()

    if cli.data:
        BASE_DIR = Path(cli.data)
    if cli.out:
        OUTPUT_DIR = Path(cli.out)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if cli.probe:
        PROBE_DIR = Path(cli.probe)
    if cli.epochs:
        NUM_EPOCHS = cli.epochs

    if cli.skip_audit:
        run_dataset_audit = lambda strict=True: True  # noqa: E731
    run_training(force=cli.force)
