# Technical Report: CNN-Based Fruit Recognition and Price Estimation System

**Course:** Deep Learning  
**Project:** Product Recognition Pipeline  
**Author:** [Student Name]  

---

## 1. Abstract
This project presents a comprehensive pipeline for automated fruit recognition and price estimation using Convolutional Neural Networks (CNNs). By leveraging five distinct state-of-the-art architectures—AlexNet, VGG-16, GoogLeNet, ResNet-50, and EfficientNet-B4—the system achieves near-perfect classification accuracy on standardized datasets. Additionally, a web-based graphical user interface (GUI) was developed to demonstrate real-time inference, category prediction, and automatic cost calculation based on a curated price database.

## 2. Introduction
In the modern retail and grocery industry, automated checkout systems and inventory management rely heavily on Computer Vision. The primary challenge is accurately identifying products and retrieving their corresponding prices without manual intervention. This project aims to solve a subset of this problem: identifying single fruit products from images and estimating their total cost.

The system is designed to:
1.  **Classify** images into 17 specific product categories.
2.  **Evaluate** the performance of multiple deep learning architectures.
3.  **Calculate** individual and combined prices.
4.  **Provide** a user-friendly web interface with a shopping cart demonstration.

## 3. Dataset Preparation
### 3.1 Data Source
The system utilizes 17 categories from the **Fruits-360** dataset to exceed the minimum requirement of 15-20.

**Selected Categories (17):**
- Apple, Banana, Cherry, Grape, Kiwi, Lemon, Mango, Orange, Peach, Pear, Pineapple, Plum, Pomegranate, Strawberry, Tomato, Potato, Onion.

### 3.2 Preprocessing and Augmentation
To ensure high model performance and robustness, several preprocessing steps were applied:
- **Resizing:** All images were resized to **224x224 pixels**, the standard input resolution for most ImageNet-pretrained models.
- **Normalization:** Pixel values were normalized using the ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
- **Data Augmentation (Training Set Only):**
    - Random Horizontal and Vertical Flips.
    - Random Rotations (up to 15 degrees).
    - Color Jitter (adjusting brightness, contrast, and saturation by 30%).

### 3.3 Data Splitting
The dataset was split into three non-overlapping sets:
- **Training (80%):** Used for adjusting model weights.
- **Validation (10%):** Used for hyperparameter tuning and early stopping.
- **Test (10%):** Used for final evaluation to represent real-world performance on unseen data.

---

## 4. Methodology: Model Implementation
Five different CNN architectures were implemented using **Transfer Learning**. Transfer learning allows us to benefit from features (edges, textures, shapes) already learned by models trained on the massive ImageNet dataset.

### 4.1 AlexNet
- **Architecture:** 5 Convolutional layers followed by 3 Fully Connected layers.
- **Role:** Serves as a baseline model due to its historical significance and simpler structure.

### 4.2 VGG-16
- **Architecture:** 16 layers with small (3x3) filters used throughout the network.
- **Role:** Demonstrates the effectiveness of deeper networks with uniform architecture.

### 4.3 GoogLeNet (Inception v1)
- **Architecture:** Utilizes "Inception modules" which perform convolutions of different sizes (1x1, 3x3, 5x5) in parallel.
- **Role:** High parameter efficiency.

### 4.4 ResNet-50
- **Architecture:** Implements "Skip Connections" (Residual blocks) to allow gradients to flow more easily through deep layers.
- **Role:** Highly recommended for preventing the vanishing gradient problem in deep networks.

### 4.5 EfficientNet-B4
- **Architecture:** Uses a compound scaling method that balances network depth, width, and resolution.
- **Role:** Represents the most advanced model in this project, optimized for both accuracy and efficiency.

---

## 5. Training Strategy
- **Optimizer:** Adam (Learning Rate = 0.001).
- **Loss Function:** Cross-Entropy Loss.
- **Hardware:** Training was performed on **Google Colab** using an **NVIDIA T4 GPU** to accelerate computations.
- **Callbacks:** 
    - **Early Stopping:** Training stopped if validation loss didn't improve for 5 epochs.
    - **LR Scheduler:** Reduced the learning rate if the model hit a plateau.

---

## 6. Model Comparison and Evaluation
After training all models for 10 epochs, a comprehensive evaluation was performed on the test set.

### 6.1 Performance Table
| Model | Acc (Top-1) | Acc (Top-5) | Price MAE | Speed (ms/img) | Size (MB) |
|-------|-------------|-------------|-----------|----------------|-----------|
| **ALEXNET** | 100.0% | 100.0% | 0.00 | 0.06 | 217.7 |
| **VGG-16** | 100.0% | 100.0% | 0.00 | 0.06 | 512.4 |
| **GOOGLENET**| 100.0% | 100.0% | 0.00 | 0.34 | 45.9 |
| **RESNET-50**| 100.0% | 100.0% | 0.00 | 0.22 | 90.1 |
| **EFFICIENTNET**| 100.0% | 100.0% | 0.00 | 0.75 | 67.8 |

### 6.2 Metrics Discussion
- **Accuracy:** All models achieved **100% accuracy** on the test set. This is primarily because the Fruits-360 dataset consists of fruits on a plain white background, making the classification task relatively straightforward for modern CNNs.
- **Price Accuracy:** Since classification was perfect, the **Mean Absolute Error (MAE)** for price estimation was 0.00, meaning the system correctly retrieved every price from the database.
- **Inference Speed:** AlexNet and VGG were the fastest per-image, while EfficientNet-B4, being more complex, took slightly longer (0.75 ms).
- **Model Size:** VGG-16 is significantly larger (512 MB) compared to GoogLeNet (45 MB), making GoogLeNet better for edge deployment despite similar accuracy.

---

## 7. Analysis: Real-World Performance
During manual testing with images from the internet, a **Domain Gap** was observed. While the models perform perfectly on the test set (white background), they may misclassify fruit when:
1.  **Complex Backgrounds:** The model was never trained on backgrounds other than white and may mistake table textures for fruit skin.
2.  **Varieties:** The model was trained on specific varieties (e.g., Yellow Apple). It might struggle with Green or Red apples if those specific colors weren't prominent in the training subset.

**Conclusion:** For a production-ready system, "Fine-tuning" on a more diverse dataset (with varied backgrounds) would be necessary.

---

## 8. Web Application
A minimal GUI was developed using **Streamlit**.
- **Features:** Sidebar model selection, single-image upload, prediction display, confidence score, and price lookup.
- **Dynamic Visualization:** Includes a bar chart showing the "Top-5" alternative predictions by the model.

---

## 9. Conclusion
This project successfully implemented a full deep learning pipeline for product recognition. By comparing five CNN architectures, we demonstrated that even simpler models like AlexNet can achieve extremely high accuracy on controlled datasets, while more complex models like EfficientNet provide better feature extraction. The integration of a price database and a web GUI provides a clear demonstration of how CNNs can be applied to solve practical retail problems.

## 10. Deliverables
- **Code:** Python scripts for training, evaluation, and GUI.
- **Weights:** 5 `.pth` files in the `saved_models/` folder.
- **Results:** Comparison CSV and Confusion Matrices.
- **Documentation:** This technical report.
