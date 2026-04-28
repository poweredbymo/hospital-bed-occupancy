

# Hospital Occupancy Prediction API

A FastAPI-powered inference service for the hospital bed occupancy ML model.
Input operational data for any service and get a utilization prediction with SHAP-based explanations.

## Setup

```bash
# 1. Save model artifacts::
import joblib
joblib.dump(grid_search_final.best_estimator_, 'models/occupancy_model.pkl')
joblib.dump(experimental_features_2, 'models/feature_cols.pkl')

# 2. Install dependencies
pip install -r app/requirements.txt

# 3. Run the server (from project root)
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in your browser.

![alt text](<Screenshot 2026-04-28 at 19.40.57.png>)

## Project Structure
