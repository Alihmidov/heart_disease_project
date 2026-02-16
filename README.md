# Heart Disease Prediction Project

This is an End-to-End Machine Learning project. I built a model to predict heart disease risk, created an API with **FastAPI**, and deployed it using **Docker** on **Render**.

**Live API Link:** [https://heart-disease-project-2-o3au.onrender.com/docs](https://heart-disease-project-2-o3au.onrender.com/docs)

## How I built this project

### 1. Data Exploration & Cleaning
* **No Encoding/Imputation:** The dataset was already numeric, so I didn't need to convert text to numbers. There were also no missing values, so no imputation was required.
* **Handling Outliers:** I identified outliers in `trestbps` and `chol` using the IQR method. I used **RobustScaler** to make the model more stable against these outliers.
* **Visuals:** I used `Seaborn` and `Matplotlib` to analyze data distributions and correlations.

### 2. Model Selection
I tested three different models to find the best predictor:
* **Logistic Regression:** Used as the baseline model.
* **Random Forest (Final Choice):** This model provided the best balance between accuracy and reliability.
* **XGBoost:** I tested XGBoost, but it was not selected because it tends to **overfit** on small datasets (this project has 302 rows) and performed slightly worse than Random Forest.

### 3. Engineering the Pipeline
* **Feature Selection:** I removed features like `fbs` and `restecg` because they had low impact on the results. This made the model simpler and more efficient.
* **Pipelines:** I integrated scaling and modeling into a single **Scikit-learn Pipeline**. This ensures consistent preprocessing and prevents **data leakage**.

## Model Results & Evaluation

In medical projects, **Recall** is critical because we must minimize missing actual cases of disease.

### Classification Report
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Healthy** | 0.89 | 0.78 | 0.83 |
| **Disease** | 0.79 | **0.90** | 0.84 |
| **Overall Accuracy** | | | **0.84** |

### Why these numbers matter:
* **High Recall (0.90):** The model correctly identifies **90% of people with heart disease**. This reduces the risk of telling a sick person they are healthy (False Negatives).
* **Confidence:** When the model predicts someone is healthy, it is correct **89% of the time** (Precision for Class 0).
* **Confusion Matrix:** The model only missed 3 cases of disease in the test set.

### 4. Stability Check (Cross-Validation)
I performed 5-fold Cross-Validation to ensure the model's performance is consistent across different data samples:
* **CV Scores:** `[0.84, 0.80, 0.90, 0.80, 0.78]`
* **Average CV Accuracy:** **0.8245 (82.5%)**
This proves the model is stable and not just performing well by chance on the test set.

## 🛠 Tech Stack
* **Python 3.12**
* **FastAPI** (Web Framework)
* **Docker** (Containerization)
* **Scikit-learn** (Machine Learning)
* **Render** (Cloud Hosting)

## How to run it locally

1. **Clone the project:**
   ```bash
   git clone [https://github.com/Alihmidov/heart_disease_project.git](https://github.com/Alihmidov/heart_disease_project.git)
   cd heart_disease_project