# Breast Cancer Detection Model

A comprehensive machine learning solution for breast cancer detection using multiple algorithms including Random Forest, XGBoost, CatBoost, LightGBM, SVM, and Neural Networks.

## Features

- **Multiple Model Comparison**: Evaluates 6 different machine learning algorithms
- **Automatic Best Model Selection**: Selects the best performing model based on F1-score
- **FastAPI Deployment**: Production-ready REST API for real-time predictions
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and cross-validation metrics

## Project Structure

```
Breast Cancer Detection/
├── Breast_cancer_dataset.csv    # Dataset
├── train_models.py              # Model training and comparison script
├── app.py                       # FastAPI application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train Models and Select Best Model

Run the training script to compare all models and select the best one:

```bash
python train_models.py
```

This script will:
- Load and preprocess the breast cancer dataset
- Train 6 different models:
  - Random Forest
  - XGBoost
  - CatBoost
  - LightGBM
  - Support Vector Machine (SVM)
  - Neural Network
- Evaluate each model using multiple metrics
- Select the best model based on F1-score
- Save the best model, scaler (if needed), and metadata

The script will generate:
- `best_model.pkl` - The best trained model
- `scaler.pkl` - Feature scaler (if required by the model)
- `model_metadata.pkl` - Model metadata and results

### Step 2: Start the FastAPI Server

Once the model is trained, start the API server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### Step 3: Make Predictions

#### Using the Interactive API Docs

1. Navigate to http://localhost:8000/docs
2. Click on the `/predict` endpoint
3. Click "Try it out"
4. Enter the feature values (or use the example values)
5. Click "Execute"

#### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Using Python

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    # ... (all other features)
}

response = requests.post(url, json=data)
print(response.json())
```

## API Endpoints

### GET `/`
Root endpoint with API information.

### GET `/health`
Health check endpoint. Returns model status.

### POST `/predict`
Single prediction endpoint.
- **Input**: JSON with all 30 feature values
- **Output**: Prediction (Benign/Malignant), probability, and confidence level

### POST `/predict/batch`
Batch prediction endpoint for multiple samples.
- **Input**: List of JSON objects with feature values
- **Output**: List of predictions

## Model Evaluation Metrics

The training script evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: Correctness of positive predictions
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: 5-fold CV for robust evaluation

## Dataset

The dataset includes 30 features derived from digitized images of fine needle aspirates (FNA) of breast masses. The target variable is:
- **B (Benign)**: Non-cancerous
- **M (Malignant)**: Cancerous

## Notes

- The dataset is automatically split into 80% training and 20% testing
- Models requiring feature scaling (SVM, Neural Network) are automatically scaled
- The best model is selected based on F1-score to balance precision and recall
- All models are trained with appropriate hyperparameters and random seeds for reproducibility

## License

This project is for educational and research purposes.

