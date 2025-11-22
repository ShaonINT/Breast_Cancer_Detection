"""
FastAPI Application for Breast Cancer Detection
Deploys the best trained model for real-time predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Detection API",
    description="API for predicting breast cancer diagnosis using machine learning",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and metadata
try:
    model = joblib.load('best_model.pkl')
    metadata = joblib.load('model_metadata.pkl')
    scaler = None
    if metadata.get('needs_scaler', False):
        try:
            scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            print("Warning: Scaler file not found but model requires scaling")
except FileNotFoundError:
    model = None
    metadata = None
    scaler = None
    print("Warning: Model files not found. Please train the model first using train_models.py")

# Feature names from metadata or default
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

if metadata and 'features' in metadata:
    FEATURE_NAMES = metadata['features']


class PredictionInput(BaseModel):
    """Input model for prediction request."""
    radius_mean: float = Field(..., description="Mean radius")
    texture_mean: float = Field(..., description="Mean texture")
    perimeter_mean: float = Field(..., description="Mean perimeter")
    area_mean: float = Field(..., description="Mean area")
    smoothness_mean: float = Field(..., description="Mean smoothness")
    compactness_mean: float = Field(..., description="Mean compactness")
    concavity_mean: float = Field(..., description="Mean concavity")
    concave_points_mean: float = Field(..., description="Mean concave points")
    symmetry_mean: float = Field(..., description="Mean symmetry")
    fractal_dimension_mean: float = Field(..., description="Mean fractal dimension")
    radius_se: float = Field(..., description="Radius standard error")
    texture_se: float = Field(..., description="Texture standard error")
    perimeter_se: float = Field(..., description="Perimeter standard error")
    area_se: float = Field(..., description="Area standard error")
    smoothness_se: float = Field(..., description="Smoothness standard error")
    compactness_se: float = Field(..., description="Compactness standard error")
    concavity_se: float = Field(..., description="Concavity standard error")
    concave_points_se: float = Field(..., description="Concave points standard error")
    symmetry_se: float = Field(..., description="Symmetry standard error")
    fractal_dimension_se: float = Field(..., description="Fractal dimension standard error")
    radius_worst: float = Field(..., description="Worst radius")
    texture_worst: float = Field(..., description="Worst texture")
    perimeter_worst: float = Field(..., description="Worst perimeter")
    area_worst: float = Field(..., description="Worst area")
    smoothness_worst: float = Field(..., description="Worst smoothness")
    compactness_worst: float = Field(..., description="Worst compactness")
    concavity_worst: float = Field(..., description="Worst concavity")
    concave_points_worst: float = Field(..., description="Worst concave points")
    symmetry_worst: float = Field(..., description="Worst symmetry")
    fractal_dimension_worst: float = Field(..., description="Worst fractal dimension")

    class Config:
        schema_extra = {
            "example": {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095,
                "texture_se": 0.9053,
                "perimeter_se": 8.589,
                "area_se": 153.4,
                "smoothness_se": 0.006399,
                "compactness_se": 0.04904,
                "concavity_se": 0.05373,
                "concave_points_se": 0.01587,
                "symmetry_se": 0.03003,
                "fractal_dimension_se": 0.006193,
                "radius_worst": 25.38,
                "texture_worst": 17.33,
                "perimeter_worst": 184.6,
                "area_worst": 2019.0,
                "smoothness_worst": 0.1622,
                "compactness_worst": 0.6656,
                "concavity_worst": 0.7119,
                "concave_points_worst": 0.2654,
                "symmetry_worst": 0.4601,
                "fractal_dimension_worst": 0.1189
            }
        }


class PredictionOutput(BaseModel):
    """Output model for prediction response."""
    prediction: str = Field(..., description="Predicted diagnosis: 'Benign (B)' or 'Malignant (M)'")
    probability: float = Field(..., description="Probability of malignant diagnosis")
    confidence: str = Field(..., description="Confidence level")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Breast Cancer Detection API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": metadata.get('model_name', None) if metadata else None
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Predict breast cancer diagnosis based on input features.
    
    Returns:
    - prediction: "Benign (B)" or "Malignant (M)"
    - probability: Probability of malignant diagnosis (0-1)
    - confidence: Confidence level (Low/Medium/High)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using train_models.py"
        )
    
    try:
        # Convert input to array
        input_dict = input_data.dict()
        features = np.array([[input_dict[feature] for feature in FEATURE_NAMES]])
        
        # Scale if needed
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        prediction_proba = model.predict_proba(features)[0]
        prediction_class = model.predict(features)[0]
        
        # Map prediction to label
        # 0 = Benign (B), 1 = Malignant (M)
        malignant_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        
        diagnosis = "Malignant (M)" if prediction_class == 1 else "Benign (B)"
        
        # Determine confidence level
        if malignant_prob >= 0.8 or malignant_prob <= 0.2:
            confidence = "High"
        elif malignant_prob >= 0.7 or malignant_prob <= 0.3:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "prediction": diagnosis,
            "probability": float(malignant_prob),
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(inputs: List[PredictionInput]):
    """
    Predict breast cancer diagnosis for multiple samples at once.
    
    Returns list of predictions for each input sample.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using train_models.py"
        )
    
    try:
        results = []
        
        for input_data in inputs:
            # Convert input to array
            input_dict = input_data.dict()
            features = np.array([[input_dict[feature] for feature in FEATURE_NAMES]])
            
            # Scale if needed
            if scaler is not None:
                features = scaler.transform(features)
            
            # Make prediction
            prediction_proba = model.predict_proba(features)[0]
            prediction_class = model.predict(features)[0]
            
            malignant_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            diagnosis = "Malignant (M)" if prediction_class == 1 else "Benign (B)"
            
            # Determine confidence level
            if malignant_prob >= 0.8 or malignant_prob <= 0.2:
                confidence = "High"
            elif malignant_prob >= 0.7 or malignant_prob <= 0.3:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            results.append({
                "prediction": diagnosis,
                "probability": float(malignant_prob),
                "confidence": confidence
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

