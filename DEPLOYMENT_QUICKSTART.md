# Quick Deployment Guide

## 🚀 Fastest Deployment Options

### 1. Render.com (Recommended - Easiest)

**Steps:**
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: breast-cancer-detection
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Click "Create Web Service"
6. Done! Your app will be live in ~2 minutes

**Note:** Make sure `best_model.pkl` and `model_metadata.pkl` are committed to your repo or train them during build.

---

### 2. Railway.app (Also Easy)

**Steps:**
1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Python and deploys
5. Done! Your app is live

---

### 3. Docker (Any Platform)

**Build and run locally:**
```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

**Deploy to any cloud that supports Docker:**
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Instances
- DigitalOcean App Platform

---

### 4. Heroku

**Quick Deploy:**
```bash
# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# Open
heroku open
```

**Make sure you have:**
- ✅ Procfile (already created)
- ✅ requirements.txt (already created)
- ✅ Model files in repo OR train during build

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] Model is trained: `python train_models.py`
- [ ] Model files exist: `best_model.pkl`, `model_metadata.pkl`
- [ ] All dependencies in `requirements.txt`
- [ ] `app.py` handles PORT environment variable (✅ done)
- [ ] Test locally: `python app.py`

---

## 🔧 Post-Deployment

1. **Test health endpoint:**
   ```
   https://your-app-url.com/health
   ```

2. **Test prediction:**
   ```
   https://your-app-url.com/docs
   ```

3. **Access web interface:**
   ```
   https://your-app-url.com
   ```

---

## 🐛 Troubleshooting

**Model not loading?**
- Ensure model files are in the repo or train during build
- Check file paths in logs

**Port errors?**
- Make sure `app.py` uses `PORT` env variable (✅ already done)

**Dependencies missing?**
- Verify `requirements.txt` is complete

---

## 📚 Detailed Guides

For detailed deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md)

---

## 🎯 Recommended: Render.com

**Why Render?**
- ✅ Free tier available
- ✅ Easy GitHub integration
- ✅ Automatic HTTPS
- ✅ Simple configuration
- ✅ No credit card required for free tier

**Steps:**
1. Push code to GitHub
2. Connect to Render
3. Deploy
4. Done!

Your app will be live at: `https://your-app-name.onrender.com`

