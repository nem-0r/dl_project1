# 🍎 Product Recognizer & Price Estimation System

A complete Deep Learning pipeline for recognizing fruits and vegetables and estimating their total cost. Developed as a CNN project for the Deep Learning course.

## 🚀 Overview
This system identifies **17 different product categories** using 5 state-of-the-art CNN architectures. It includes a web application for real-time inference and a "Shopping Cart" feature to calculate the combined price of multiple detected items.

## 📂 Project Structure
```
fruit_project/
├── FINAL_TECHNICAL_REPORT.md  # 📄 8-15 Page Report (Methodology & Results)
├── app.py                     # 🌐 Web UI (Streamlit)
├── download_dataset.py        # 📥 Dataset downloader & preprocessor
├── dataset.py                 # 🖼️ PyTorch DataLoaders & Transforms
├── train.py                   # 🧠 Training script for all 5 models
├── evaluate.py                # 📊 Comparison & Metrics script
├── prices.json                # 💰 Price database (17 categories)
├── requirements.txt           # 📦 Python dependencies
├── models/                    # 🏗️ Model Architectures
│   ├── alexnet.py, vgg.py, googlenet.py, resnet.py, efficientnet.py
└── saved_models/              # 💾 Trained Weights & Confusion Matrices
```

## 🛠️ Setup & Installation

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download & Split Dataset:**
   ```bash
   python3 download_dataset.py
   ```
   *Note: This downloads a subset of the Fruits-360 dataset (17 classes: Apple, Banana, Potato, Onion, etc.) and splits it 80/10/10.*

## 🧠 Training & Evaluation

To train all 5 models from scratch:
```bash
python3 train.py --model efficientnet --epochs 10  # Best performing
# ... repeat for alexnet, vgg, googlenet, resnet
```

To run a full comparison and generate the metrics table:
```bash
python3 evaluate.py
```
*Outputs Accuracy (Top-1/5), Precision, Recall, F1, Model Size, Speed, and Price MAE.*

## 💻 Web App Demonstration

Launch the interactive interface:
```bash
python3 -m streamlit run app.py
```

**Features:**
- Choose between any the 5 pre-trained models.
- Upload images to see classification and confidence.
- **Add to Bill:** Sum up prices of multiple items to calculate **Total Cost**.

## 📊 Evaluation Results Summary
All models were evaluated on the held-out test set. **EfficientNet-B4** provided the best balance of accuracy and inference speed. Deep details are available in the [FINAL_TECHNICAL_REPORT.md](FINAL_TECHNICAL_REPORT.md).

## 📄 Deliverables Compliance
- [x] 15-20 Categories (17 implemented)
- [x] 5 Architectures (VGG, ResNet, etc.)
- [x] Top-1 & Top-5 Accuracy
- [x] Price Evaluation (MAE/RMSE)
- [x] Web GUI with Total Cost calculator
- [x] Documentation & Technical Report
