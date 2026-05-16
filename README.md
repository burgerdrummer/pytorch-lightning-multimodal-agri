# 🌾 Multimodal Agricultural Diagnostic System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-792EE5.svg?logo=pytorch-lightning&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=Kaggle&logoColor=white)

An AI-driven diagnostic system that predicts crop diseases by simultaneously analyzing visual foliage data (images) and localized environmental metrics (tabular weather data). 

Most agricultural diagnostic models rely entirely on computer vision. However, visual symptoms like brown spots can be caused by both deadly fungal infections and benign sun damage. This project implements an **Early Fusion Multimodal Architecture**, allowing the model to weigh visual symptoms against historical rainfall and temperature data to produce a highly contextual, accurate diagnosis.

---

## 🧠 Architecture Overview

This project utilizes a bifurcated input stream that converges into a unified fully connected network:

1. **The Visual Pipeline (CNN):** High-resolution leaf imagery ($224 \times 224$) is passed through a custom Convolutional Neural Network to extract spatial features and pathological anomalies.
2. **The Environmental Pipeline (Tabular):** Localized agronomic data (Rainfall, Temperature, Yield) is min-max scaled to prevent gradient explosions.
3. **Early Fusion:** The flattened spatial feature vector is concatenated with the normalized tabular vector.
4. **Classification:** The unified tensor is passed through dense feed-forward layers with Dropout to predict a multi-class probability distribution across various crop diseases.

---

## 🛠️ Tech Stack

* **Framework:** PyTorch & PyTorch Lightning (for scalable, reproducible training loops)
* **Metrics & Evaluation:** TorchMetrics, Scikit-Learn, Grad-CAM
* **Data Processing:** Pandas, NumPy
* **Environment:** Google Colab (optimized for fast Disk I/O and GPU acceleration)

---

## 📊 Datasets

This project fuses two separate datasets from Kaggle to create a realistic multimodal environment:
* **Visual Data:** [20,000+ Multi-Class Crop Disease Images](https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images)
* **Contextual Data:** [Crop Production India with Weather Features](https://www.kaggle.com/datasets/atharvpratapsingh/crop-production-india-with-weather-features)

*Note: To optimize training speed, an $O(1)$ dictionary lookup was engineered to map specific image directories to their corresponding environmental data, reducing preprocessing time from hours to seconds.*

---

## 🚀 How to Run (Google Colab)

1. Open the `.ipynb` notebook in Google Colab.
2. Ensure your Runtime is set to T4 GPU (Runtime > Change runtime type).
3. Set your Kaggle API token in the environment variables block:

   import os
   os.environ['KAGGLE_API_TOKEN'] = "YOUR_TOKEN_HERE"

4. Run the notebook sequentially. The script will automatically download the datasets, engineer the data fusion registry, and commence PyTorch Lightning training.

---

## 🔬 Evaluation Framework & Interpretability

Because agricultural datasets are highly skewed (e.g., significantly more images of healthy crops than specific localized mutations), performance evaluation must extend beyond simple accuracy. This project implements a rigorous, multi-tiered framework to verify model robustness, measure per-class reliability, and combat dataset bias.

### 1. Core Metrics & Classification Report
The model utilizes Precision, Recall, and the F1-Score across all distinct classes to ensure the network does not disproportionately favor the majority class. A live Seaborn-based Confusion Matrix is plotted to trace exactly which pathologies the network might visually confuse.

| Metric | Mathematical Definition | Strategic Implication for Diagnostic Models |
| :--- | :--- | :--- |
| **Precision** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ | Measures the reliability of a positive disease diagnosis, critical for preventing unnecessary, costly pesticide application. |
| **Recall** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ | Measures the ability to detect the disease when present, crucial for preventing devastating crop loss due to missed infections. |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Provides a harmonic mean, balancing the trade-off between precision and recall in highly imbalanced datasets. |

### 2. Modality Ablation Study (The "Blindfold" Test)
To mathematically prove that the tabular environmental data is actively contributing to the diagnostics rather than being ignored, an ablation test is performed during inference. The notebook passes a zero-tensor (`torch.zeros_like`) to isolate and evaluate the performance degradation when the model is forced to make a decision using *images only*.

### 3. Grad-CAM Visual Interpretability
To move away from "black-box" machine learning, the notebook implements Gradient-weighted Class Activation Mapping (Grad-CAM). By wrapping the multimodal model and tracking the gradients at the final convolutional layer (`model.conv3`), the system overlays a visual heatmap on the leaf. This allows human operators to verify that the network is focusing on actual foliage lesions rather than background noise like soil or sky.

### 4. High-Confidence Error Analysis
A model that is wrong but highly certain is a liability in production. The notebook dynamically tracks the Softmax probability distributions of incorrect validation outputs. It isolates and ranks the top 5 most confident mistakes (`incorrect_mask`), exposing underlying labeling anomalies or edge cases within the Kaggle datasets.

## 🔮 Future Enhancements

* Hyperparameter Tuning: Integrating Optuna to automate the optimization of learning rates, batch sizes, and dropout layers.
* Streamlit Deployment: Building an interactive frontend where farmers can upload photos and adjust weather sliders for real-time inference.

---
*Developed by @burgerdrummer*
