# 📌 Airline Passenger Satisfaction Prediction 

Live App: [Click Here](https://airline-passenger-satisfaction-classifier-hungrychaos.streamlit.app/)

## 🚀 Project Overview

This project is an end-to-end machine learning classification system built using a modular pipeline architecture. It covers data preprocessing, model training, evaluation, and deployment through a Streamlit web application for real-time predictions.

The goal of this project is to simulate a production-like ML workflow with clean code separation and reproducibility.

---

## 🧠 Problem Statement

The airline industry operates in a highly competitive environment where customer satisfaction directly influences revenue, loyalty, and brand positioning. While airlines invest heavily in operational efficiency and inflight services, they often lack a reliable data-driven system to understand what actually drives passenger satisfaction.

---

## 🏗️ Project Architecture

```
project/
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── model.pkl
│
├── app/
│   └── app.py   # Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

1. **Data Ingestion**

   * Load dataset and perform basic validation

2. **Preprocessing**

   * Handle missing values
   * Encode categorical variables
   * Feature scaling (if required)

3. **Feature Engineering**

   * Transform raw inputs into model-ready features

4. **Model Training**

   * Train classification model (e.g., Logistic Regression / Random Forest / XGBoost)
   * Evaluate using accuracy, precision, recall

5. **Model Saving**

   * Save trained model using `joblib/pickle`

6. **Inference**

   * Load model and make predictions via `predict.py`

7. **Deployment**

   * Streamlit app for real-time user input and prediction

---

## 🖥️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Streamlit
* Joblib / Pickle

---

## 📊 Model Performance


* Accuracy: 96%
* Precision: 97%
* Recall: 93%
* F1 Score: 95%

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/HungryChaos/Airline-Passenger-Satisfaction-Classifier.git
cd Airline-Passenger-Satisfaction-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional if already trained)

```bash
python src/train.py
```

### 4. Run Streamlit app

```bash
streamlit run app/app.py
```

---

## 📌 Key Features

* Modular ML pipeline (production-style structure)
* Reusable preprocessing & training components
* Clean separation of training and inference
* Interactive Streamlit UI for predictions
* Easy to extend for new models or datasets

---

## 📁 Future Improvements

* Add CI/CD pipeline for model training
* Deploy on cloud (AWS / Render / HuggingFace Spaces)
* Add experiment tracking (MLflow)
* Hyperparameter tuning automation

---
