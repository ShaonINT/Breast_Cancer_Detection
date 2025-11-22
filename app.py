"""
FastAPI Application for Breast Cancer Detection
Deploys the best trained model for real-time predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import os

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
import os

model = None
metadata = None
scaler = None

try:
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
    metadata_path = os.path.join(os.path.dirname(__file__), 'model_metadata.pkl')
    
    if os.path.exists(model_path) and os.path.exists(metadata_path):
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        
        if metadata and metadata.get('needs_scaler', False):
            scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                print("Warning: Scaler file not found but model requires scaling")
        
        print(f"✓ Model loaded successfully: {metadata.get('model_name', 'Unknown') if metadata else 'Unknown'}")
    else:
        print(f"Warning: Model files not found.")
        print(f"  Looking for: {model_path}")
        print(f"  Looking for: {metadata_path}")
        if not os.path.exists(model_path):
            print(f"  ✗ best_model.pkl not found")
        if not os.path.exists(metadata_path):
            print(f"  ✗ model_metadata.pkl not found")
        print("Please train the model first using train_models.py")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    import traceback
    traceback.print_exc()
    model = None
    metadata = None
    scaler = None

# Feature order from metadata or default
# Note: Pydantic model uses underscores (concave_points_mean), CSV uses spaces (concave points_mean)
# Models only care about feature order, not names
FEATURE_ORDER = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# If metadata exists, use the feature order from there (but convert spaces to underscores for Pydantic)
if metadata and 'features' in metadata:
    # Create mapping: CSV feature names (with spaces) -> Pydantic field names (with underscores)
    csv_feature_names = metadata['features']
    feature_mapping = {}
    for csv_name in csv_feature_names:
        pydantic_name = csv_name.replace(' ', '_')
        feature_mapping[pydantic_name] = csv_name
    # Use the order from metadata
    FEATURE_ORDER = [name.replace(' ', '_') for name in csv_feature_names]


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


# Set up static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - serves the web interface."""
    static_file = os.path.join(static_dir, "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {
        "message": "Breast Cancer Detection API",
        "version": "1.0.0",
        "endpoints": {
            "web_interface": "/",
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


@app.get("/example", tags=["Prediction"])
async def get_random_example():
    """
    Get a random example from the dataset.
    Returns a random sample that can be used for testing predictions.
    """
    import random
    import pandas as pd
    
    try:
        # Load dataset
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Breast_cancer_dataset.csv'))
        
        # Drop id and diagnosis columns
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        if 'diagnosis' in df.columns:
            df = df.drop('diagnosis', axis=1)
        
        # Normalize column names (replace spaces with underscores)
        df.columns = [col.replace(' ', '_') for col in df.columns]
        
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Get a random row
        random_row = df.sample(n=1).iloc[0]
        
        # Convert to dictionary with proper naming (Pydantic format)
        example_dict = {}
        for col in df.columns:
            # Convert CSV column names to Pydantic field names
            pydantic_name = col.replace(' ', '_')
            example_dict[pydantic_name] = float(random_row[col])
        
        return example_dict
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading example: {str(e)}")


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
        # Extract values in the order expected by the model
        features = np.array([[input_dict[feature] for feature in FEATURE_ORDER]])
        
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
            # Extract values in the order expected by the model
            features = np.array([[input_dict[feature] for feature in FEATURE_ORDER]])
            
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
    # Get port from environment variable (for cloud deployments) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

