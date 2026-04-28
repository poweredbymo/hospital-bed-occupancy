from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np 
import shap 
import os

app = FastAPI(title = 'Hospital Bed Occupancy Predictor')

artifact_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hospital_utilization_model_v1.pkl')

artifact = joblib.load(artifact_path)
model = artifact['model']
features = artifact['features']

# SHAP explainer
explainer = shap.TreeExplainer(model)

# static file to be mounted 
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

class PredictionRequest(BaseModel):
    service: str 
    staff_morale : float
    avg_los: float 
    utilization_lag1: float
    event: str = 'none' 
    week: int = 53 
    staff_on_duty: int = 20 

@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.post("/predict")
def predict (req: PredictionRequest):
    input_data= {col: 0.0 for col in features}
    input_data.update({
        'week': req.week,
        'staff_morale': req.staff_morale,
        'avg_los': req.avg_los,
        'utilization_lag1': req.utilization_lag1,
        'staff_on_duty': req.staff_on_duty,
    })

    # One-hot encode service and event 
    service_key = f'service_{req.service}'
    event_key = f'event_{req.event}'

    if service_key in input_data:
        input_data[service_key] = 1.0
    if event_key in input_data:
        input_data[event_key] = 1.0

    X = pd.DataFrame([input_data])[features]

    pred = float(np.clip(model.predict(X)[0], 0, 1))

    # labelling 
    if pred >= 0.90:
        risk = "Critical"
        risk_color = "#ef4444"
    elif pred >= 0.75:
        risk = "High"
        risk_color = "#f97316"
    elif pred >= 0.50:
        risk = "Moderate"
        risk_color = "#eab308"
    else:
        risk = "Normal"
        risk_color = "#d0fadf"

    # SHAP explanations (top drivers)
    shap_values = explainer.shap_values(X)
    shap_series = pd.Series(shap_values[0], index=features)
    top_drivers = (
        shap_series
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )

    drivers = []
    for feat in top_drivers.index:
        impact = float(shap_series[feat])
        value = float(X[feat].iloc[0])
        # label user friendly labels
        label = (feat
                 .replace('service_', 'Service: ')
                 .replace('event_', 'Event: ')
                 .replace('utilization_lag1', 'Last week occupancy')
                 .replace('avg_los', 'Average length of stay')
                 .replace('staff_morale', 'Staff morale')
                 .replace('staff_on_duty', 'Staff on duty')
                 .replace('week', 'Week number')
                 .replace('_', ' ').title())
        drivers.append({
            "feature": label,
            "impact": round(impact, 4),
            "value": round(value, 3),
            "direction": "up" if impact > 0 else "down"
        })

    return {
            'utilization': round(pred,4),
            'utilization_pct': f"{pred:.1%}",
            'risk': risk,
            'risk_color': risk_color,
            'drivers': drivers
        }
    
@app.get('/health')
def health():
    return {"status": "OK"}