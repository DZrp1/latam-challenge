from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import List
from joblib import load
import pandas as pd
import inspect
import os
from .model import DelayModel

app = FastAPI()

# Pydantic models
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class FlightPayload(BaseModel):
    flights: List[Flight]

@app.on_event("startup")
def load_model_and_metadata():
    """Load the DelayModel instance at startup and store it in app.state."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    artifact_path = os.path.join(project_root, 'models', 'trained_model.joblib')

    if not os.path.exists(artifact_path):
        raise RuntimeError(f"Model artifact not found at {artifact_path}. Please run the training script first.")

    app.state.model = load(artifact_path)
    if not app.state.model._is_fitted:
        raise RuntimeError("Loaded model is not fitted. Please run the training script.")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    frame = inspect.currentframe()
    request_found = None
    try:
        while frame:
            if 'request' in frame.f_locals and isinstance(frame.f_locals['request'], Request):
                request_found = frame.f_locals['request']
                break
            frame = frame.f_back
    finally:
        del frame

    if not request_found:
        raise HTTPException(status_code=500, detail="Could not retrieve request from context.")

    try:
        body = await request_found.json()
        payload = FlightPayload(**body)
    except (ValidationError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    model: DelayModel = app.state.model

    if not payload.flights:
        raise HTTPException(status_code=400, detail="No flights provided for prediction.")

    for flight in payload.flights:
        if flight.OPERA not in model.airlines:
            raise HTTPException(status_code=400, detail=f"Invalid airline: {flight.OPERA}")
        if flight.TIPOVUELO not in model.types:
            raise HTTPException(status_code=400, detail=f"Invalid flight type: {flight.TIPOVUELO}")
        if flight.MES not in model.months:
            raise HTTPException(status_code=400, detail=f"Invalid month: {flight.MES}")

    features_df = pd.DataFrame([f.dict() for f in payload.flights])
    preprocessed_features = model.preprocess(data=features_df)
    predictions = model.predict(features=preprocessed_features)
    
    return {"predict": predictions}