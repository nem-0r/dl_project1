"""
download_dataset.py

Downloads the Fruits-360 dataset from the official GitHub repository.
No Kaggle API key required!

Usage:
    python download_dataset.py
"""

import os
import zipfile
import shutil
import requests
from tqdm import tqdm
import random

# ==============================================================
# CONFIG
# ==============================================================
GITHUB_URL = (
    "https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip"
)
ZIP_PATH = "fruits360_raw.zip"
EXTRACT_DIR = "fruits360_raw"
DATA_DIR = "data"

# 15 classes we want to keep (must match folder names in the dataset)
CLASSES = [
    "Apple Golden 1",
    "Banana",
    "Cherry 1",
    "Grape Blue",
    "Kiwi",
    "Lemon",
    "Mango",
    "Orange",
    "Peach",
    "Pear",
    "Pineapple",
    "Plum 1",
    "Pomegranate",
    "Strawberry",
    "Tomato 1",
    "Potato Red",
    "Onion Red",
]

# Friendly names for our 15 classes (must match prices.json keys)
CLASS_RENAMES = {
    "Apple Golden 1": "Apple",
    "Banana": "Banana",
    "Cherry 1": "Cherry",
    "Grape Blue": "Grape",
    "Kiwi": "Kiwi",
    "Lemon": "Lemon",
    "Mango": "Mango",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pear": "Pear",
    "Pineapple": "Pineapple",
    "Plum 1": "Plum",
    "Pomegranate": "Pomegranate",
    "Strawberry": "Strawberry",
    "Tomato 1": "Tomato",
    "Potato Red": "Potato",
    "Onion Red": "Onion",
}

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

random.seed(42)


def download_file(url: str, dest: str):
    print(f"Downloading dataset from GitHub...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Downloaded to {dest}")


def extract_zip(zip_path: str, extract_dir: str):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print("Extraction complete.")


def create_splits(raw_train_dir: str, output_dir: str):
    """Split images into train/val/test folders."""
    print("Creating train/val/test splits...")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    found_classes = []
    for original_name, friendly_name in CLASS_RENAMES.items():
        src_folder = os.path.join(raw_train_dir, original_name)
        if not os.path.isdir(src_folder):
            print(f"  [WARN] Class folder not found: {src_folder}, skipping.")
            continue

        images = [
            f
            for f in os.listdir(src_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, files in splits.items():
            dest_class_dir = os.path.join(output_dir, split, friendly_name)
            os.makedirs(dest_class_dir, exist_ok=True)
            for fname in files:
                shutil.copy(
                    os.path.join(src_folder, fname),
                    os.path.join(dest_class_dir, fname),
                )

        found_classes.append(friendly_name)
        print(
            f"  {friendly_name:15s}: {len(splits['train'])} train | "
            f"{len(splits['val'])} val | {len(splits['test'])} test"
        )

    print(f"\nDone! Found {len(found_classes)} classes: {found_classes}")
    return found_classes


def main():
    # Step 1: Download
    if not os.path.exists(ZIP_PATH):
        download_file(GITHUB_URL, ZIP_PATH)
    else:
        print(f"{ZIP_PATH} already exists, skipping download.")

    # Step 2: Extract
    if not os.path.exists(EXTRACT_DIR):
        extract_zip(ZIP_PATH, EXTRACT_DIR)
    else:
        print(f"{EXTRACT_DIR} already exists, skipping extraction.")

    # Step 3: Find the Training folder inside extracted directory
    raw_train_dir = None
    for root, dirs, _ in os.walk(EXTRACT_DIR):
        if "Training" in dirs:
            raw_train_dir = os.path.join(root, "Training")
            break

    if raw_train_dir is None:
        raise FileNotFoundError(
            "Could not find 'Training' folder in extracted dataset."
        )
    print(f"Found Training folder at: {raw_train_dir}")

    # Step 4: Create splits
    if os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}' folder already exists. Delete it to re-split.")
    else:
        create_splits(raw_train_dir, DATA_DIR)

    # Step 5: Cleanup
    print("\nCleaning up temporary files...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)

    print("\n✅ Dataset ready in ./data/train  ./data/val  ./data/test")
    print("   Run: python train.py --model resnet --epochs 10")


if __name__ == "__main__":
    main()
