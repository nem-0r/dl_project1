"""
dataset.py

Builds train/val/test DataLoaders from data/ directory.
Also exports CLASS_NAMES list used by models and app.
"""

import os
import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PRICES_FILE = Path(__file__).parent / "prices.json"
IMG_SIZE = 224   # standard for most pre-trained models
BATCH_SIZE = 32
NUM_WORKERS = 0  # set to 0 for Mac compatibility

# ----------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

INFER_TRANSFORMS = VAL_TRANSFORMS   # same as val — no augmentation


# ----------------------------------------------------------------
# Helper: load prices
# ----------------------------------------------------------------
def load_prices() -> dict:
    with open(PRICES_FILE, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------------
def get_dataloaders(data_dir: str = None, batch_size: int = BATCH_SIZE):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    class_names is a sorted list of fruit category names.
    """
    root = Path(data_dir) if data_dir else DATA_DIR

    train_dataset = datasets.ImageFolder(
        root / "train", transform=TRAIN_TRANSFORMS
    )
    val_dataset = datasets.ImageFolder(
        root / "val", transform=VAL_TRANSFORMS
    )
    test_dataset = datasets.ImageFolder(
        root / "test", transform=VAL_TRANSFORMS
    )

    class_names = train_dataset.classes  # sorted list of folder names

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
    )

    return train_loader, val_loader, test_loader, class_names


# ----------------------------------------------------------------
# Single-image inference transform
# ----------------------------------------------------------------
def prepare_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Transform a PIL image to a batched tensor ready for model inference.
    Returns shape [1, 3, 224, 224].
    """
    tensor = INFER_TRANSFORMS(pil_image.convert("RGB"))
    return tensor.unsqueeze(0)   # add batch dimension
