# Customer Churn Prediction System

A production-ready machine learning project that predicts whether a telecom customer is likely to churn using **Logistic Regression with Scikit-Learn**.
The model is deployed as a **FastAPI REST API**, containerized with **Docker**, and hosted live on **Render**.

---

# Live Demo

API Base URL:
https://customer-churn-prediction-using-logistic-r109.onrender.com/

Interactive API Docs:
https://customer-churn-prediction-using-logistic-r109.onrender.com/docs

---

# Project Overview

Customer churn is a major business problem for subscription-based companies.
This project helps identify customers who are likely to leave so the business can take preventive action.

The system accepts customer information as input and returns:

* Churn Prediction (`0 = No`, `1 = Yes`)
* Churn Probability Score

---

# Features

* End-to-end machine learning pipeline
* Data cleaning and preprocessing
* Feature engineering
* Logistic Regression model training
* Probability-based churn prediction
* FastAPI REST API
* Auto-generated Swagger documentation
* Dockerized deployment
* Cloud hosting on Render

---

# Tech Stack

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn

## Backend

* FastAPI
* Uvicorn
* Pydantic

## Deployment

* Docker
* Render

---

# Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── feature_engineering.py
│
├── models/
│   └── churn_model.pkl
│
├── requirements.txt
├── Dockerfile
└── README.md
```

# Machine Learning Workflow

## 1. Data Loading

Loaded telecom customer churn dataset.

## 2. Data Cleaning

* Converted data types
* Handled missing values
* Removed invalid records

## 3. Feature Engineering

Created additional useful features such as customer spending patterns.

## 4. Preprocessing

* StandardScaler for numeric columns
* OneHotEncoder for categorical columns

## 5. Model Training

Used Logistic Regression with pipeline architecture.

## 6. Evaluation

Measured:

* Accuracy
* ROC AUC Score
* Confusion Matrix
* Classification Report

## 7. Deployment

Saved trained model and exposed it through FastAPI.

---

# API Endpoints

## Root Endpoint

```http
GET /
```

Response:

```json
{
  "message": "Churn Prediction API Running",
  "version": "1.0.0"
}
```

---

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Predict Churn

```http
POST /predict
```

Sample Request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1020.3
}
```

Sample Response:

```json
{
  "prediction": 1,
  "probability": 0.78,
  "risk_level": "High"
}
```

---

# Run Locally

## 1. Clone Repository

```bash
git clone Customer-Churn-Prediction-using-Logistic-Regression.git
cd Customer-Churn-Prediction-using-Logistic-Regression
```

## 2. Create Virtual Environment

```bash
py  -3.10 venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run FastAPI App

```bash
uvicorn app.main:app --reload
```

## 5. Open Browser

```text
http://127.0.0.1:8000/docs
```

---

# Run with Docker

## Build Image

```bash
docker build -t churn-api .
```

## Run Container

```bash
docker run -p 8000:8000 churn-api
```

---

# Business Impact

This solution can help businesses:

* Reduce customer churn
* Improve retention strategies
* Prioritize high-risk customers
* Increase long-term revenue

---

# Future Improvements

* XGBoost / Random Forest comparison
* Streamlit frontend dashboard
* CI/CD pipeline
* Model monitoring
* Authentication and rate limiting

---

# Author

Pooja

---

# If You Found This Useful

Please star the repository and connect with me.
