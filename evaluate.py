"""
evaluate.py  —  Evaluate and compare all trained models on the test set.

Usage:
    python evaluate.py

Outputs:
    - Per-model: accuracy, precision, recall, F1, inference speed, model size
    - Confusion matrix plot for each model
    - Full comparison table printed to console
"""

import os
import json
import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from dataset import get_dataloaders
from models import MODEL_REGISTRY

SAVED_MODELS_DIR = "saved_models"


def load_model(model_name: str, num_classes: int, device: torch.device):
    path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(path):
        print(f"  [SKIP] {model_name}: no saved weights at {path}")
        return None

    model = MODEL_REGISTRY[model_name](num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def get_top_k_accuracy(outputs, labels, k=5):
    with torch.no_grad():
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item()


def get_predictions_and_metrics(model, loader, device, prices, class_names):
    all_preds, all_labels = [], []
    top5_correct = 0
    total = 0
    
    price_errors = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Top-1
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Top-5
            k = min(5, len(class_names))
            top5_correct += get_top_k_accuracy(outputs, labels, k=k)
            total += labels.size(0)
            
            # Price estimation metrics (MAE/RMSE)
            for true_idx, pred_idx in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                true_name = class_names[true_idx]
                pred_name = class_names[pred_idx]
                
                true_price = prices.get(true_name, 0)
                pred_price = prices.get(pred_name, 0)
                
                price_errors.append(pred_price - true_price)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    price_errors = np.array(price_errors)
    
    mae = np.mean(np.abs(price_errors))
    rmse = np.sqrt(np.mean(price_errors**2))
    top5_acc = top5_correct / total
    
    return all_labels, all_preds, top5_acc, mae, rmse


def measure_inference_speed(model, loader, device, n_batches=10):
    """Returns average inference time per image in milliseconds."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            start = time.perf_counter()
            _ = model(images)
            elapsed = time.perf_counter() - start
            times.append(elapsed / images.size(0) * 1000)  # ms per image
    return np.mean(times)


def get_model_size_mb(model_name: str) -> float:
    path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def plot_confusion_matrix(labels, preds, class_names, model_name):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")
    plt.tight_layout()
    out_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Confusion matrix saved → {out_path}")


def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # Data & Prices
    from dataset import load_prices
    prices = load_prices()
    _, _, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Classes: {class_names}\n")

    results = []

    for model_name in MODEL_REGISTRY.keys():
        print(f"{'='*50}")
        print(f"Evaluating: {model_name.upper()}")

        model = load_model(model_name, num_classes, device)
        if model is None:
            continue

        labels, preds, top5_acc, p_mae, p_rmse = get_predictions_and_metrics(
            model, test_loader, device, prices, class_names
        )

        acc        = accuracy_score(labels, preds)
        precision  = precision_score(labels, preds, average="weighted", zero_division=0)
        recall     = recall_score(labels, preds, average="weighted", zero_division=0)
        f1         = f1_score(labels, preds, average="weighted", zero_division=0)
        speed_ms   = measure_inference_speed(model, test_loader, device)
        size_mb    = get_model_size_mb(model_name)

        print(f"  Accuracy (Top-1): {acc:.4f}")
        print(f"  Accuracy (Top-5): {top5_acc:.4f}")
        print(f"  Precision:        {precision:.4f}")
        print(f"  Recall:           {recall:.4f}")
        print(f"  F1-score:         {f1:.4f}")
        print(f"  Price MAE:        {p_mae:.2f}")
        print(f"  Price RMSE:       {p_rmse:.2f}")
        print(f"  Inference:        {speed_ms:.2f} ms/image")
        print(f"  Size:             {size_mb:.1f} MB")

        print("\n  Per-class report:")
        print(classification_report(labels, preds, target_names=class_names,
                                    zero_division=0))

        plot_confusion_matrix(labels, preds, class_names, model_name)

        results.append({
            "Model":       model_name.upper(),
            "Acc Top-1":   f"{acc:.4f}",
            "Acc Top-5":   f"{top5_acc:.4f}",
            "Price MAE":   f"{p_mae:.2f}",
            "F1-score":    f"{f1:.4f}",
            "Speed ms":    f"{speed_ms:.2f}",
            "Size MB":     f"{size_mb:.1f}",
        })

    # ── Comparison Table ──────────────────────────────────────
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        csv_path = os.path.join(SAVED_MODELS_DIR, "comparison_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nTable saved → {csv_path}")
    else:
        print("No trained models found. Run train.py first.")


if __name__ == "__main__":
    main()
