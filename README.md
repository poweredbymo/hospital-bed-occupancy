# Hospital Bed Occupancy Predictor 
An end-to-end Machine Learning solution designed to forecast hospital utilization rates 12 weeks in advance. By moving beyond naive historical averages, this project integrates operational data (staffing, morale, and service types) to provide Explainable AI (XAI) insights for healthcare administrators.

## Project Overview
Traditional hospital forecasting often relies on "persistence" models (assuming next week looks like last week). This project implements a Gradient Boosting Regressor that identifies the underlying drivers of occupancy, allowing for proactive capacity management during operational shocks.

## Tech Stack
* **Language:** Python 3.9
* **Modeling:** Scikit-Learn (Gradient Boosting Regressor), Pandas, NumPy
* **Interpretability:** SHAP (TreeExplainer)
* **API Framework:** FastAPI, Uvicorn, Pydantic
* **DevOps**: Docker

---

## Data Science Methodology

### 1. Data Processing & Engineering
The raw dataset was transformed into a structured time-series format through the following engineering steps:
* **Calculated Length of Stay:** Derived patient-level metrics via (Departure Date - Arrival Date).
* **Daily Occupancy:** Calculated by tracking overlapping patient date ranges to determine active bed counts.
* **Weekly Aggregation:** Aggregated daily metrics and staffing data into a weekly grain to align with institutional planning cycles.
* **Target Variable:** Defined **Utilization Rate** as the ratio of occupied beds to total capacity (scaled $0.0$ to $1.0$).
* **Temporal Features:** Engineered `utilization_lag1` (previous week's rate) and cyclical week numbers to capture seasonal trends.

### 2. Eliminating Data Leakage
During initial development, the model achieved an unrealistic **MAE of 0.001**.
* **Diagnosis:** Post-event features (data only available *after* the target week ends) had leaked into the training set.
* **The Fix:** I performed a "Leakage Audit," pruning features to ensure only $T-0$ (pre-event) data was available for training. This resulted in an **Honest Operational Model** with an MAE of **0.1067**.

### 3. Model Training & Optimization
* **Time-Series Split:** To respect chronological integrity, I trained on the **first 40 weeks** and validated on the remaining weeks (no random shuffling).
* **Hyperparameter Tuning:** Conducted a **GridSearchCV** on the Gradient Boosting Regressor, optimizing `learning_rate`, `max_depth`, and `n_estimators` to handle non-linear hospital dynamics without overfitting.

---

## Evaluation & Insights

### Service-Level Performance
The model reveals that hospital volatility is not distributed equally:
| Service | MAE (Error) | Insight |
| :--- | :--- | :--- |
| **Surgery** | `0.1695` | Highly volatile, dependent on elective scheduling shifts. |
| **ICU** | `0.1401` | High error due to unpredictable emergency surges. |
| **General Med** | `0.0863` | Stable, characterized by longer, predictable stays. |
| **Emergency** | `0.0309` | Highly predictable; driven by consistent historical patterns. |

### Residual Analysis
Analysis of the residuals showed a **"Ceiling Effect"** near 100% utilization. The model tends to be slightly optimistic, acting as a high-water mark for capacity planning while sometimes missing sudden, sharp drops in occupancy.

---


# Deployment and API Architecture 

The model is served via FastAPI-powered inference application. 
Input operational data for any service and get a utilization prediction with SHAP-based explanations on top 5 drivers of the prediction. 

## Setup & Deployment 

### Save Model Artifacts

```python
import joblib
joblib.dump({'model': final_model, 'features': experimental_features_2}, 'models/hospital_utilization_model_v1.pkl')
```

### Local Deployment 

```bash
# 1. Install dependencies
pip install -r app/requirements.txt

# 2. Run the server (from project root)
uvicorn app.main:app --reload
```

### Docker Deployment 

```bash
# 1.Build the image
docker build -t occupancy-predictor .

# 2. Run the container
docker run -p 8000:8000 occupancy-predictor
```

Then open http://localhost:8000 in your browser and input operational data as seen below:

![alt text](<Screenshot 2026-04-28 at 19.40.57.png>)

## Project Structure

```text
├── app/
│   ├── main.py          # FastAPI server & SHAP logic
│   ├──requirements.txt 
│   └── static/          
│       └── index.html        
├── models/
│   └── hospital_utilization_model_v1.pkl     
├── notebooks/
│   ├── 01_eda.ipynb     # Initial data exploration and feature engineering
│   └── 02_modeling.ipynb # Model training, evaluation and packaging   
└── Dockerfile