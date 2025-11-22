# 🏥 Breast Cancer Detection System

A comprehensive machine learning solution for breast cancer detection with a user-friendly web interface. This project compares multiple state-of-the-art algorithms (Random Forest, XGBoost, CatBoost, LightGBM, SVM, and Neural Networks) and provides real-time predictions through an intuitive web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/License-Educational%20Use-yellow.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Web Interface](#-web-interface)
- [API Documentation](#-api-documentation)
- [Model Comparison](#-model-comparison)
- [Understanding the Results](#-understanding-the-results)
- [Deployment](#-deployment)
- [Dataset Information](#-dataset-information)
- [Medical Disclaimer](#-medical-disclaimer)
- [Contributing](#-contributing)
- [License](#-license)

---
<div align="center">
  <a href="https://breast-cancer-detection-34ui.onrender.com/">
    <img src="https://img.shields.io/badge/DEMO-Live_App-FF4B4B?style=for-the-badge&logo=appveyor" alt="Live Demo" />
  </a>
  
  <br />
  <br />

  <img src="https://github.com/ShaonINT/Breast_Cancer_Detection/blob/main/static/screenshot.png?raw=true" alt="Breast Cancer Detection Interface" width="800" style="border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);" />

  <p><em>Interactive Interface demonstrating Probability Scoring and Feature Explanation</em></p>
</div>

---
## ✨ Features

### 🎯 Core Features

- **📊 Multiple Model Comparison**: Evaluates 6 different machine learning algorithms
  - Random Forest
  - XGBoost
  - CatBoost
  - LightGBM
  - Support Vector Machine (SVM)
  - Neural Network

- **🎯 Automatic Best Model Selection**: Selects the best performing model based on F1-score

- **🚀 Production-Ready API**: FastAPI-based REST API for real-time predictions

- **💻 User-Friendly Web Interface**: 
  - Beautiful, responsive web UI
  - Interactive form with all 30 features
  - Real-time predictions
  - Color-coded results (Green for Benign, Red for Malignant)

- **📚 Educational Tooltips**: 
  - Help icons (?) next to each field
  - Plain-language explanations for all medical terminology
  - Tooltips on hover with detailed descriptions
  - Clear explanations of Malignant vs Benign

- **🎲 Random Example Generation**: 
  - Load random examples from the dataset
  - Different example each time
  - Pre-filled form for quick testing

- **📈 Comprehensive Evaluation**: 
  - Accuracy, Precision, Recall, F1-score
  - 5-fold Cross-Validation
  - Detailed classification reports
  - Confusion matrix analysis

- **🔧 Deployment Ready**: 
  - Docker support
  - Cloud deployment guides (Render, Heroku, Railway, AWS, GCP, Azure)
  - Traditional hosting deployment instructions

---

## 📁 Project Structure

```
Breast Cancer Detection/
├── Breast_cancer_dataset.csv      # Dataset with 569 samples and 30 features
├── train_models.py                # Model training and comparison script
├── app.py                         # FastAPI application with web interface
├── requirements.txt               # Python dependencies
├── static/
│   └── index.html                # User-friendly web interface
├── Dockerfile                     # Docker container configuration
├── docker-compose.yml             # Docker Compose setup
├── Procfile                       # Heroku deployment configuration
├── render.yaml                    # Render.com deployment configuration
├── runtime.txt                    # Python version specification
├── deploy.sh                      # Quick deployment script
├── setup_hosting.sh               # Hosting deployment script
├── start_server.sh                # Server start script
├── passenger_wsgi.py              # WSGI entry point for hosting providers
├── DEPLOYMENT.md                  # Comprehensive deployment guide
├── DEPLOYMENT_QUICKSTART.md       # Quick deployment guide
├── DEPLOYMENT_HOSTING.md          # Traditional hosting deployment guide
├── RENDER_DEPLOYMENT.md           # Render.com specific guide
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ShaonINT/Breast_Cancer_Detection.git
cd Breast_Cancer_Detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas
- numpy
- scikit-learn
- xgboost
- catboost
- lightgbm
- fastapi
- uvicorn
- pydantic
- joblib

---

## ⚡ Quick Start

### 1. Train the Models

```bash
python train_models.py
```

This will:
- Load and preprocess the dataset
- Train all 6 models
- Evaluate each model
- Select the best model (highest F1-score)
- Save model files (`best_model.pkl`, `model_metadata.pkl`)

**Expected Output:**
```
================================================================================
BREAST CANCER DETECTION - MODEL COMPARISON
================================================================================

Training and evaluating models...

Random Forest:    Accuracy: 0.9649, F1-Score: 0.9500
XGBoost:          Accuracy: 0.9737, F1-Score: 0.9630
CatBoost:         Accuracy: 0.9561, F1-Score: 0.9367
LightGBM:         Accuracy: 0.9649, F1-Score: 0.9500
SVM:              Accuracy: 0.9737, F1-Score: 0.9630
Neural Network:   Accuracy: 0.9737, F1-Score: 0.9630

Best Model: XGBoost (F1-Score: 0.9630)
```

### 2. Start the Web Application

```bash
python app.py
```

Then open your browser:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📖 Usage Guide

### Using the Web Interface

1. **Navigate to** http://localhost:8000

2. **Read the Information Box** at the top to understand:
   - What the measurements mean
   - What Malignant and Benign mean
   - How to use the tool

3. **Fill in the Form**:
   - Hover over the **?** icon next to any field for explanations
   - Or click **"Load Random Example"** to fill with sample data
   - Enter all 30 feature values

4. **Get Prediction**:
   - Click **"Get Prediction"** button
   - View results:
     - **B** (Benign) in green = Non-cancerous
     - **M** (Malignant) in red = Cancerous
     - Probability percentage
     - Confidence level (High/Medium/Low)

5. **Try Another**:
   - Click **"Load Random Example"** for a different sample
   - Or **"Clear Form"** to start fresh

### Understanding the Form Fields

Each field has:
- **Label**: The feature name
- **? Icon**: Hover for detailed explanation
- **Help Text**: Brief description below the label
- **Input Field**: Enter the numeric value

**Field Categories:**
- **Mean Features**: Average measurements across all cells (10 fields)
- **Standard Error Features**: Variation/uncertainty in measurements (10 fields)
- **Worst Features**: Most abnormal/severe measurements (10 fields)

---

## 🌐 Web Interface Features

### 🎨 User-Friendly Design

- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Color-Coded Results**: 
  - Green background for Benign (B) results
  - Red background for Malignant (M) results
- **Clear Visual Hierarchy**: Organized sections for easy navigation

### 📚 Educational Features

- **Interactive Tooltips**: Hover over ? icons for explanations
- **Plain Language**: All medical terms explained in simple language
- **Info Box**: Overview of measurements and terminology at the top
- **Result Explanations**: Each prediction includes explanation of what it means

### 🎲 Example Features

- **Random Examples**: Get different examples each time
- **Real Data**: Examples come from actual dataset
- **Pre-filled Form**: One-click to populate all fields

---

## 🔌 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. GET `/`
Returns the web interface HTML page.

#### 2. GET `/health`
Health check endpoint to verify the API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "XGBoost"
}
```

#### 3. GET `/example`
Get a random example from the dataset with proper precision formatting.

**Response:**
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  ...
}
```

#### 4. POST `/predict`
Single prediction endpoint.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": "Malignant (M)",
  "probability": 0.95,
  "confidence": "High"
}
```

#### 5. POST `/predict/batch`
Batch prediction for multiple samples at once.

**Request Body:**
```json
[
  {
    "radius_mean": 17.99,
    ...
  },
  {
    "radius_mean": 13.54,
    ...
  }
]
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "Malignant (M)",
      "probability": 0.95,
      "confidence": "High"
    },
    {
      "prediction": "Benign (B)",
      "probability": 0.12,
      "confidence": "High"
    }
  ],
  "count": 2
}
```

### Using the API with Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    # ... all other features
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']*100:.2f}%")
print(f"Confidence: {result['confidence']}")
```

### Using the API with cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    ...
  }'
```

---

## 📊 Model Comparison

### Supported Models

1. **Random Forest**: Ensemble learning with decision trees
2. **XGBoost**: Gradient boosting framework
3. **CatBoost**: Gradient boosting with categorical features support
4. **LightGBM**: Fast gradient boosting framework
5. **SVM (Support Vector Machine)**: Kernel-based classification
6. **Neural Network**: Multi-layer perceptron classifier

### Evaluation Metrics

Each model is evaluated using:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of positive (malignant) predictions
- **Recall**: Ability to find all malignant cases
- **F1-Score**: Harmonic mean of precision and recall (used for model selection)
- **Cross-Validation**: 5-fold CV for robust performance estimation

### Best Model Selection

The model with the **highest F1-score** is automatically selected as the best model. F1-score balances precision and recall, making it ideal for medical diagnosis where both false positives and false negatives are important.

---

## 📖 Understanding the Results

### Prediction Output

The model returns one of two predictions:

#### 🟢 Benign (B) - Non-Cancerous

- **Meaning**: The cells appear normal and healthy
- **Result**: The tissue is **NOT cancerous**
- **Display**: Shown with green color
- **What to do**: Continue regular monitoring as recommended by your doctor

#### 🔴 Malignant (M) - Cancerous

- **Meaning**: The cells show abnormalities that may indicate cancer
- **Result**: The tissue **may be cancerous**
- **Display**: Shown with red color
- **What to do**: **Consult a healthcare professional immediately** for proper medical evaluation and diagnosis

### Confidence Levels

- **High**: Probability ≥ 80% or ≤ 20% (very confident prediction)
- **Medium**: Probability between 70-80% or 20-30% (moderately confident)
- **Low**: Probability between 30-70% (less confident, may require more tests)

### Probability

The probability value indicates how confident the model is that the sample is malignant (M). For example:
- **0.95** = 95% probability of being malignant
- **0.12** = 12% probability of being malignant (88% chance it's benign)

---

## 🚀 Deployment

### Quick Deployment Options

#### Option 1: Render.com (Recommended - Easiest)

1. Go to [render.com](https://render.com) and sign up
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `ShaonINT/Breast_Cancer_Detection`
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Click **"Create Web Service"**
6. Your app will be live in ~2 minutes!

📖 **Detailed Guide**: See [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)

#### Option 2: Railway.app

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Railway auto-detects and deploys
4. Done!

#### Option 3: Docker

```bash
docker build -t breast-cancer-detection .
docker run -p 8000:8000 breast-cancer-detection
```

#### Option 4: Traditional Web Hosting

See [DEPLOYMENT_HOSTING.md](./DEPLOYMENT_HOSTING.md) for detailed instructions.

### Deployment Guides

- **Quick Start**: [DEPLOYMENT_QUICKSTART.md](./DEPLOYMENT_QUICKSTART.md)
- **Comprehensive Guide**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Traditional Hosting**: [DEPLOYMENT_HOSTING.md](./DEPLOYMENT_HOSTING.md)
- **Render.com**: [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)

---

## 📊 Dataset Information

### Dataset Source

**Kaggle Dataset**: [Breast Cancer Dataset](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset)

The dataset is available on Kaggle and is based on the Wisconsin Breast Cancer Database (WBCD). This dataset is widely used in machine learning research for breast cancer classification tasks.

### Dataset Description

The dataset contains features computed from digitized images of **Fine Needle Aspirate (FNA)** samples of breast masses. FNA is a diagnostic procedure where a thin needle is used to extract a small sample of cells from a breast mass for microscopic examination.

**Key Characteristics:**
- **Source**: Features are derived from images of cell nuclei obtained through FNA procedures
- **Purpose**: Classify breast masses as benign (non-cancerous) or malignant (cancerous)
- **Medical Context**: This is a binary classification problem critical for early breast cancer detection
- **Data Quality**: Well-curated dataset with minimal missing values

### Dataset Statistics

- **Total Samples**: 569 instances
- **Features**: 30 numerical features
- **Target Distribution**:
  - **B (Benign)**: 357 cases (62.7%) - Non-cancerous tissue
  - **M (Malignant)**: 212 cases (37.3%) - Cancerous tissue
- **Class Imbalance**: Slight imbalance toward benign cases (typical in medical datasets)

### Feature Categories

The 30 features are organized into three categories, each measuring 10 different characteristics of cell nuclei:

1. **Mean Features** (10 features): Average measurements across all cells in the image
   - Provides overall characterization of cell nuclei

2. **Standard Error Features** (10 features): Standard error (variation) in measurements
   - Indicates consistency and variability across cells

3. **Worst Features** (10 features): Largest (worst/most abnormal) measurements found
   - Captures the most severe abnormalities in cell nuclei

### Feature Descriptions

Each of the 10 measured characteristics provides different insights into cell structure:

- **Radius**: Distance from center to points on the perimeter (size indicator)
- **Texture**: Standard deviation of gray-scale values (surface appearance variation)
- **Perimeter**: Distance around the boundary of the cell nucleus
- **Area**: Size of the cell nucleus (often larger in malignant cells)
- **Smoothness**: Local variation in radius lengths (boundary smoothness)
- **Compactness**: Perimeter² / (Area - 1) (how circular/compact the shape is)
- **Concavity**: Severity of concave portions of the contour (indentations)
- **Concave Points**: Number of concave portions (frequency of indentations)
- **Symmetry**: How symmetrical the cell nucleus is (normal cells are more symmetrical)
- **Fractal Dimension**: Complexity of the boundary ("coastline approximation")

### Clinical Significance

This dataset is particularly valuable because:
- **Early Detection**: Enables classification based on cell characteristics visible in FNA samples
- **Non-Invasive**: FNA is less invasive than surgical biopsies
- **Fast Results**: Can provide quicker preliminary diagnosis
- **Pattern Recognition**: Machine learning models can identify subtle patterns not easily visible to the human eye

### Data Preprocessing

In this project, the dataset is preprocessed by:
- Normalizing feature names (removing spaces)
- Encoding target variable (M=1, B=0)
- Handling missing values (if any)
- Splitting into training (80%) and testing (20%) sets with stratification

**Note**: For detailed explanations of each feature, hover over the **?** icons in the web interface!

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This tool is for **educational and research purposes only**. 

- ❌ **NOT a substitute** for professional medical diagnosis
- ❌ **NOT intended** for clinical decision-making
- ✅ **ALWAYS consult** with a qualified healthcare provider
- ✅ Results should be interpreted by medical professionals

**This application does not provide medical advice. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.**

---

## 🔧 Technical Details

### Model Training Process

1. **Data Preprocessing**:
   - Remove ID column
   - Normalize feature names (spaces to underscores)
   - Remove unnamed/empty columns
   - Encode target variable (M=1, B=0)
   - Handle missing values

2. **Data Splitting**:
   - 80% training set
   - 20% test set
   - Stratified split to maintain class distribution

3. **Feature Scaling**:
   - Automatic scaling for SVM and Neural Network
   - Other models use raw features

4. **Model Evaluation**:
   - 5-fold cross-validation
   - Multiple metrics (Accuracy, Precision, Recall, F1)
   - Best model selection based on F1-score

### Model Files

After training, the following files are generated:

- `best_model.pkl`: The selected best model (joblib format)
- `model_metadata.pkl`: Metadata including:
  - Model name
  - Feature names and order
  - Performance metrics
  - Whether scaler is needed
- `scaler.pkl`: Feature scaler (if required by the model)

---

## 🛠️ Development

### Running Tests

```bash
# Test the API
python app.py
# Then visit http://localhost:8000/health

# Test predictions
curl http://localhost:8000/predict -X POST -H "Content-Type: application/json" -d '{...}'
```

### Project Structure Details

- **`train_models.py`**: Model comparison and training pipeline
- **`app.py`**: FastAPI application with endpoints and model loading
- **`static/index.html`**: Complete web interface with explanations
- **Deployment files**: Docker, Render, Heroku, hosting configurations

---

## 📝 Usage Examples

### Example 1: Web Interface

1. Start the server: `python app.py`
2. Open browser: http://localhost:8000
3. Click "Load Random Example"
4. Click "Get Prediction"
5. View result: **M** (Malignant) or **B** (Benign)

### Example 2: API Programmatic Access

```python
import requests

# Get a random example
example = requests.get('http://localhost:8000/example').json()

# Make a prediction
prediction = requests.post(
    'http://localhost:8000/predict',
    json=example
).json()

print(f"Result: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']}")
```

### Example 3: Batch Predictions

```python
import requests

examples = [
    requests.get('http://localhost:8000/example').json()
    for _ in range(5)
]

results = requests.post(
    'http://localhost:8000/predict/batch',
    json=examples
).json()

for i, result in enumerate(results['predictions']):
    print(f"Sample {i+1}: {result['prediction']}")
```

---

## 🎯 Key Features Explained

### User-Friendly Terminology

- **Mean**: Average value across all cells
- **SE (Standard Error)**: How much values vary
- **Worst**: Most abnormal/severe value
- **Benign (B)**: Non-cancerous (healthy)
- **Malignant (M)**: Cancerous (abnormal)

### Interactive Help

Every field has:
- **Help icon (?)**: Hover to see detailed explanation
- **Brief description**: Plain language summary
- **Tooltip**: Full explanation of what the measurement means

---

## 📚 Additional Resources

### Deployment Documentation

- **[DEPLOYMENT_QUICKSTART.md](./DEPLOYMENT_QUICKSTART.md)**: Quick deployment guide
- **[DEPLOYMENT.md](./DEPLOYMENT.md)**: Comprehensive deployment options
- **[DEPLOYMENT_HOSTING.md](./DEPLOYMENT_HOSTING.md)**: Traditional hosting guide
- **[RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)**: Render.com specific guide

### Scripts

- **`deploy.sh`**: Quick deployment script
- **`setup_hosting.sh`**: Hosting setup automation
- **`start_server.sh`**: Server start script

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement

- Additional model architectures
- Enhanced feature engineering
- Better UI/UX improvements
- Documentation improvements
- Performance optimizations

---

## 📄 License

This project is for **educational and research purposes only**.

---

## 👤 Author

**Shaon Biswas**

- GitHub: [@ShaonINT](https://github.com/ShaonINT)
- Repository: [Breast_Cancer_Detection](https://github.com/ShaonINT/Breast_Cancer_Detection)

---

## 🙏 Acknowledgments

- **Dataset Source**: [Breast Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset) by [wasiqaliyasir](https://www.kaggle.com/wasiqaliyasir)
- **Original Dataset**: Wisconsin Breast Cancer Database (WBCD)
- **Libraries**: scikit-learn, XGBoost, CatBoost, LightGBM, FastAPI
- **Community**: Open source machine learning community

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the deployment guides
- Review the documentation files

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
