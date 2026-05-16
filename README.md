# 🌾 Multimodal Agricultural Diagnostic and Yield Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-792EE5.svg?logo=pytorch-lightning&logoColor=white)

The agricultural sector faces persistent challenges regarding crop yield optimization and the early detection of pathological foliage diseases. This project operationalizes an automated, artificial intelligence-driven diagnostic system capable of analyzing both visual data and environmental metrics to provide immense utility for precision agriculture and sustainable farming. 

By leveraging **Convolutional Neural Networks (CNNs)** in conjunction with multiple continuous tabular input variables, the architecture culminates in a multi-class probabilistic output. It serves as a practical application of handling multiple diverse inputs to produce a unified diagnostic classification.

---

## 🧠 Mathematical and Algorithmic Foundations

This architecture relies on a **bifurcated input stream** to process high-resolution imagery of crop foliage alongside regional environmental data.

### 1. Visual Processing Pipeline (CNN)
The primary stream processes high-resolution imagery of crop foliage to identify pathological features. Input image tensors—represented as three-dimensional matrices containing color channels, height, and width—are subjected to the following sequence:
* **Spatial Feature Extraction:** Handled via the `nn.Conv2d` module. These convolutional filters act as localized feature detectors, capturing edges, textures, and pathological anomalies within the foliage by computing the dot product between the filter weights and the localized input pixels.
* **Non-Linear Activation:** The outputs of the convolution operations (feature maps) are passed through the **Rectified Linear Unit (ReLU)** activation function, implemented as `F.relu()`. This introduces non-linearity and effectively resolves the vanishing gradient problem inherent in deep networks by outputting the input directly if it is positive, and zero otherwise.
* **Downsampling:** To mitigate the curse of dimensionality, reduce computational load, and achieve spatial invariance, Max Pooling layers (`nn.MaxPool2d`) downsample the feature maps, retaining only the most salient activations within defined kernel windows.
* **Tensor Flattening:** Following several alternating layers of convolution and pooling, the multi-dimensional tensor is flattened into a one-dimensional vector using the `torch.flatten()` operation.

### 2. Tabular Pipeline & Data Standardization
Simultaneously, the secondary tabular inputs containing environmental metrics (such as ambient temperature, historical fertilizer usage, and regional rainfall) must be standardized. Because neural networks calculate loss and gradients based on the magnitude of weights and inputs, disparate scales across variables (e.g., rainfall in millimeters versus fertilizer in kilograms) can cause severe instability during optimization. These variables are normalized using **min-max scaling** to bound the values strictly between zero and one, preventing gradients from exploding during backpropagation.

### 3. Feature Concatenation & Classification
The flattened visual feature vector and the normalized tabular vector are concatenated along the appropriate dimensional axis. This unified tensor is then routed through:
* **Fully Connected Layers (`nn.Linear`):** Where complex, non-linear relationships between the visual symptoms and the environmental conditions are modeled.
* **The SoftMax Function:** The terminal layer utilizes SoftMax to produce a probability distribution across multiple predefined crop disease classifications (e.g., *Healthy*, *Leaf Blast*, or *Brown Spot*). Mathematically, this operation involves exponentiating each output value and dividing it by the sum of all exponentiated outputs, ensuring values are strictly bounded between zero and one and sum exactly to one.

> **Note on Loss & Inference:** During the training phase, PyTorch's `nn.CrossEntropyLoss()` is utilized. This intrinsically computes the log-softmax and the negative log-likelihood, eliminating the need to manually apply the SoftMax activation within the forward pass. During the inference phase, the `torch.argmax()` function is applied directly to the raw logits to select the discrete class index associated with the highest probability.

---

## ⚡ PyTorch Lightning Implementation Mechanics

Implementing this multimodal architecture within PyTorch Lightning requires defining a custom class that inherits from `L.LightningModule`. This encapsulation ensures full compatibility with the PyTorch Lightning `Trainer` object, which automates device placement and epoch iteration. 

The implementation follows a strict structural paradigm:
* **Initialization (`__init__`):** The convolutional layers, pooling layers, and linear layers are defined as class attributes. Weights are initialized, and the random seed is fixed using `L.seed_everything(seed=42)` to ensure exact reproducibility across differing hardware environments and training runs.
* **Forward Pass (`forward`):** Defines the explicit computational graph. It routes the image tensor through the CNN layers, flattens the output, concatenates the environmental variables, and processes the unified tensor through the dense feed-forward layers.
* **Optimization Strategy (`configure_optimizers`):** This hook is specified to return the optimization algorithm. While traditional Stochastic Gradient Descent (SGD) is viable, the **Adam optimizer** (`torch.optim.Adam`) is utilized for its superior performance in complex image spaces, as it computes adaptive learning rates for each parameter based on estimates of first and second moments of the gradients.
* **Training Loop (`training_step`):** Batches of training data are unpacked into image tensors, tabular tensors, and target labels. The model computes the forward pass, and the error is quantified using Cross-Entropy Loss.

---

## 📊 Dataset Engineering and Regional Context

Developing a highly robust model requires diverse, regionally specific data. This architecture is designed to integrate real-world Indian agricultural datasets spanning multiple modalities:

1. **DeepCrop AI Project:** Provides a multimodal dataset combining **YOLOv8-based image segmentation formats** tailored specifically for Indian sugarcane farming, alongside symptom-based questionnaire data.
2. **20,000+ Multi-Class Crop Disease Dataset:** Offers publicly licensed imagery spanning multiple crop varieties and localized disease instances.
3. **Crop Yield in Indian States Dataset:** Encompasses agronomic data from 1997 to 2020, detailing variables such as season, area under cultivation, annual rainfall, and fertilizer usage across multiple states.

Merging the visual data with historical agronomic contexts forces the model to weigh environmental vulnerabilities alongside visible leaf damage.

---

## 🔬 Evaluation Framework

Performance evaluation for highly imbalanced agricultural datasets must extend beyond simple accuracy. The model utilizes precision, recall, and the F1-score across all distinct classes to ensure the network does not disproportionately favor the majority class.

| Metric | Mathematical Definition | Strategic Implication for Diagnostic Models |
| :--- | :--- | :--- |
| **Precision** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ | Indicates the reliability of a positive disease diagnosis, critical for preventing unnecessary pesticide application. |
| **Recall** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ | Indicates the ability to detect the disease when present, crucial for preventing crop loss due to missed infections. |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Provides a harmonic mean, balancing the trade-off between precision and recall in highly skewed datasets. |

---

## 🚀 Deployment Roadmap

Once fully trained within the Google Colab environment, model weights are serialized using PyTorch's native saving mechanics (`trainer.save_checkpoint`). 

The eventual deployment architecture relies on the **Streamlit web framework**, which facilitates the rapid transformation of Python scripts into interactive web applications without requiring front-end JavaScript expertise. Streamlit provides a clean, production-ready user interface where operators can upload crop images, input local weather conditions via interactive sliders, and receive real-time diagnostic predictions powered by the underlying PyTorch Lightning inference engine.

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

## 🔬 Evaluation & Interpretability

To ensure the model is robust and not just relying on dataset bias, the following rigorous evaluation methods are implemented in the notebook:

* Macro F1-Score & Confusion Matrix: Ensures the model does not disproportionately favor majority classes (e.g., healthy crops) in highly imbalanced datasets.
* Modality Ablation Study: Blinds the model to the weather data by passing a zero-tensor during inference to mathematically prove that the environmental data improves the diagnostic F1-score.
* Grad-CAM Visual Interpretability: Generates gradient-weighted heatmaps over the original input images, allowing human operators to verify that the CNN is looking at actual leaf lesions rather than background noise.
* High-Confidence Error Analysis: Extracts and isolates predictions where the model was highly confident but incorrect, highlighting areas for dataset balancing and future improvement.

---

## 🔮 Future Enhancements

* Full-Stack Observability: Routing real-time inference logs (weather payloads + model confidence) into an ELK Stack (Elasticsearch, Logstash, Kibana) to monitor for Data Drift.
* Hyperparameter Tuning: Integrating Optuna to automate the optimization of learning rates, batch sizes, and dropout layers.
* Streamlit Deployment: Building an interactive frontend where farmers can upload photos and adjust weather sliders for real-time inference.

---
*Developed by @burgerdrummer*
