Heart Disease Prediction Project

This is an End-to-End Machine Learning project. I built a model to predict heart disease risk, created an API with FastAPI, and deployed it using Docker on Render.

Live API Link: https://heart-disease-project-2-o3au.onrender.com/docs
Project Structure

The project is organized following professional software engineering practices:

    app/: Contains the FastAPI application (main.py).

    models/: Contains the trained Scikit-learn pipeline and feature list (.pkl files).

    data/: Contains raw and processed datasets.

    notebooks/: Jupyter notebooks for EDA and Model Training.

How I built this project
1. Data Exploration & Cleaning

    No Encoding/Imputation: The dataset was already numeric and had no missing values.

    Handling Outliers: Identified outliers in trestbps and chol using the IQR method. Used RobustScaler to make the model stable against these outliers.

    Visuals: Analyzed data distributions and correlations using Seaborn and Matplotlib.

2. Model Selection

I tested three different models to find the best predictor:

    Logistic Regression: Used as the baseline model.

    Random Forest (Final Choice): Provided the best balance between accuracy and reliability.

    XGBoost: Tested but not selected because it tends to overfit on small datasets (302 rows) and performed slightly worse than Random Forest.

3. Engineering the Pipeline

    Feature Selection: Removed features like fbs and restecg due to low impact.

    Pipelines: Integrated scaling and modeling into a single Scikit-learn Pipeline for consistent preprocessing and to prevent data leakage.

Model Results & Evaluation

In medical projects, Recall is critical to minimize missing actual cases (False Negatives).
Classification Report
Class	   Precision	Recall	F1-Score
Healthy	0.89	      0.78	   0.83
Disease	0.79	      0.90	   0.84
Overall Accuracy			0.84
Why these numbers matter:

    High Recall (0.90): The model correctly identifies 90% of people with heart disease.

    Stability Check: 5-fold Cross-Validation yielded an average accuracy of 82.5%, proving the model is stable.

🛠 Tech Stack

    Python 3.12

    FastAPI (Web Framework)

    Docker (Containerization)

    Scikit-learn (Machine Learning)

    Render (Cloud Hosting)

How to run it locally

    Clone the project:
    Bash

    git clone https://github.com/Alihmidov/heart_disease_project.git
    cd heart_disease_project

    Run with Docker (Recommended):
    Bash

    docker build -t heart-disease-api .
    docker run -p 10000:10000 heart-disease-api

    Run with Python (Manual):
    Bash

    python -m venv venv
    source venv/Scripts/activate  # On Linux use: source venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 10000

    Access API Documentation:
    Open http://localhost:10000/docs in your browser to test predictions.