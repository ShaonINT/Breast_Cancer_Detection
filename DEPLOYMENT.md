# Deployment Guide for Breast Cancer Detection API

This guide covers multiple deployment options for the Breast Cancer Detection FastAPI application.

## Prerequisites

1. **Train the model first** (if not already done):
```bash
python train_models.py
```

2. **Verify model files exist**:
```bash
ls -lh *.pkl
```
You should see: `best_model.pkl` and `model_metadata.pkl`

---

## Option 1: Local Deployment

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Access the application at:
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Option 2: Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t breast-cancer-detection .

# Run the container
docker run -d -p 8000:8000 --name breast-cancer-api breast-cancer-detection
```

### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up -d
```

---

## Option 3: Heroku Deployment

### Prerequisites
- Heroku CLI installed
- Heroku account

### Steps

1. **Login to Heroku**:
```bash
heroku login
```

2. **Create Heroku app**:
```bash
heroku create your-app-name
```

3. **Create Procfile**:
```
web: python app.py
```

4. **Create runtime.txt** (optional, to specify Python version):
```
python-3.10.12
```

5. **Deploy**:
```bash
git init
git add .
git commit -m "Initial commit"
git push heroku main
```

6. **Set port dynamically** (update `app.py`):
```python
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

7. **Open your app**:
```bash
heroku open
```

---

## Option 4: Render Deployment

### Steps

1. **Create account** at [render.com](https://render.com)

2. **Create new Web Service**

3. **Connect your GitHub repository**

4. **Configure settings**:
   - **Name**: breast-cancer-detection
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python train_models.py`
   - **Start Command**: `python app.py`
   - **Port**: 8000

5. **Add environment variables** (if needed)

6. **Deploy**

---

## Option 5: Railway Deployment

### Steps

1. **Install Railway CLI**:
```bash
npm i -g @railway/cli
```

2. **Login**:
```bash
railway login
```

3. **Initialize project**:
```bash
railway init
```

4. **Create railway.json**:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python app.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100
  }
}
```

5. **Deploy**:
```bash
railway up
```

---

## Option 6: Google Cloud Run

### Steps

1. **Install Google Cloud SDK**

2. **Set up project**:
```bash
gcloud config set project YOUR_PROJECT_ID
```

3. **Build and deploy**:
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/breast-cancer-api

# Deploy to Cloud Run
gcloud run deploy breast-cancer-api \
  --image gcr.io/YOUR_PROJECT_ID/breast-cancer-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

---

## Option 7: AWS Elastic Beanstalk

### Steps

1. **Install EB CLI**:
```bash
pip install awsebcli
```

2. **Initialize EB**:
```bash
eb init -p python-3.10 breast-cancer-api
```

3. **Create `.ebextensions/python.config`**:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app:app
```

4. **Deploy**:
```bash
eb create breast-cancer-api-env
eb deploy
```

---

## Option 8: Azure App Service

### Steps

1. **Install Azure CLI**

2. **Login**:
```bash
az login
```

3. **Create resource group**:
```bash
az group create --name breast-cancer-rg --location eastus
```

4. **Create App Service plan**:
```bash
az appservice plan create --name breast-cancer-plan \
  --resource-group breast-cancer-rg --sku B1 --is-linux
```

5. **Create web app**:
```bash
az webapp create --resource-group breast-cancer-rg \
  --plan breast-cancer-plan --name your-app-name \
  --runtime "PYTHON:3.10"
```

6. **Deploy**:
```bash
az webapp up --resource-group breast-cancer-rg --name your-app-name
```

---

## Important Notes for All Deployments

### 1. Model Files
Ensure model files (`best_model.pkl`, `model_metadata.pkl`, `scaler.pkl` if needed) are included in deployment:
- Add to git (if small enough) or
- Upload to cloud storage (S3, GCS, etc.) and download at startup

### 2. Environment Variables
For production, use environment variables:
```python
import os

# In app.py
DEBUG = os.environ.get("DEBUG", "False") == "True"
PORT = int(os.environ.get("PORT", 8000))
```

### 3. CORS Configuration
For production, restrict CORS origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 4. Production Server
Use production ASGI server:
```bash
# Install gunicorn with uvicorn workers
pip install gunicorn

# Run with gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 5. Security
- Use HTTPS in production
- Add authentication if needed
- Implement rate limiting
- Add API key authentication for sensitive deployments

---

## Quick Deploy Script

Create `deploy.sh`:
```bash
#!/bin/bash

echo "Training model..."
python train_models.py

echo "Checking model files..."
if [ ! -f "best_model.pkl" ]; then
    echo "Error: best_model.pkl not found!"
    exit 1
fi

echo "Starting server..."
python app.py
```

Make executable:
```bash
chmod +x deploy.sh
```

---

## Testing Deployment

After deployment, test your API:

```bash
# Health check
curl https://your-app-url.com/health

# Prediction test
curl -X POST "https://your-app-url.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"radius_mean": 17.99, ...}'
```

---

## Monitoring

### Add logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health check endpoint
Already available at `/health`

---

## Troubleshooting

1. **Model not loading**: Ensure model files are in the correct directory
2. **Port already in use**: Change port or kill existing process
3. **Dependencies missing**: Verify all packages in requirements.txt are installed
4. **Memory issues**: Increase instance size or optimize model

---

## Recommended Deployment Platforms

- **Quick & Easy**: Render, Railway, Heroku
- **Enterprise**: AWS, GCP, Azure
- **Container-based**: Docker on any cloud provider

Choose based on your needs, budget, and scale requirements!

