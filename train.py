"""
train.py  —  Unified training script for all 5 CNN models.

Usage examples:
    python train.py --model resnet --epochs 10
    python train.py --model alexnet --epochs 15 --lr 0.0005
    python train.py --model vgg --epochs 10 --batch_size 16
    python train.py --model googlenet --epochs 10
    python train.py --model efficientnet --epochs 10

After training, weights are saved to: saved_models/{model_name}_best.pth
"""

import argparse
import time
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloaders
from models import MODEL_REGISTRY

# ----------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on Fruits-360")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Train from scratch (no transfer learning)")
    return parser.parse_args()


# ----------------------------------------------------------------
# Training loop (one epoch)
# ----------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ----------------------------------------------------------------
# Validation loop
# ----------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    args = parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    # Data
    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    # Save class names mapping
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/class_names.json", "w") as f:
        json.dump(class_names, f)

    # Model
    pretrained = not args.no_pretrain
    model = MODEL_REGISTRY[args.model](num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5,
                                  verbose=True)

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    save_path = f"saved_models/{args.model}_best.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>9} {'Val Acc':>8} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6} {t_loss:>11.4f} {t_acc:>9.2%} "
              f"{v_loss:>9.4f} {v_acc:>8.2%} {current_lr:>10.6f}")

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": v_acc,
                "class_names": class_names,
            }, save_path)
            print(f"  ✅ Saved best model → {save_path}  (val_acc={v_acc:.2%})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚡ Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"   Best val accuracy: {best_val_acc:.2%}")
    print(f"   Weights saved to:  {save_path}")

    # Save training history
    history_path = f"saved_models/{args.model}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"   History saved to:  {history_path}")


if __name__ == "__main__":
    main()
