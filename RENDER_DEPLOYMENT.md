# Render.com Deployment - Fix Model Not Loaded Error

## Problem
The model files (`best_model.pkl`, `model_metadata.pkl`) are not in your GitHub repository (they're in `.gitignore`), so Render can't find them.

## Solution: Train Model During Build

You need to update your Render build command to train the model during deployment.

---

## Step 1: Update Build Command in Render

1. Go to your Render dashboard: https://dashboard.render.com
2. Click on your web service
3. Go to **Settings** tab
4. Scroll down to **Build Command**
5. Change it to:
   ```bash
   pip install -r requirements.txt && python train_models.py
   ```
6. Click **Save Changes**

---

## Step 2: Verify Start Command

Make sure your **Start Command** is:
```bash
python app.py
```

---

## Step 3: Redeploy

1. Go to **Manual Deploy** section
2. Click **Clear build cache & deploy**
3. Wait for deployment to complete (~3-5 minutes)
4. Check the build logs to verify:
   - Dependencies installed
   - Model training completed
   - Model files created

---

## Alternative Solution: Commit Model Files to Git

If training takes too long on Render, you can commit the model files:

### Option A: Temporarily allow model files

1. **Remove model files from .gitignore temporarily:**
   ```bash
   # Edit .gitignore and comment out these lines:
   # best_model.pkl
   # model_metadata.pkl
   ```

2. **Add and commit model files:**
   ```bash
   git add -f best_model.pkl model_metadata.pkl
   git commit -m "Add trained model files for deployment"
   git push origin main
   ```

3. **Update .gitignore back** (optional, to ignore future changes)

4. **Update Render build command** to just:
   ```bash
   pip install -r requirements.txt
   ```

5. **Redeploy on Render**

---

## Step 4: Verify Deployment

After deployment, check:

1. **Health endpoint:**
   ```
   https://your-app-name.onrender.com/health
   ```
   Should show:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "model_name": "XGBoost" (or whatever model was selected)
   }
   ```

2. **Test prediction:**
   ```
   https://your-app-name.onrender.com/docs
   ```

---

## Troubleshooting

### Build Fails During Model Training

**Problem:** Render build times out or runs out of memory

**Solutions:**
1. **Commit model files instead** (see Option A above)
2. **Use a smaller model** (reduce iterations in train_models.py)
3. **Upgrade Render plan** (if on free tier, there are resource limits)

### Model Files Still Not Found

**Check build logs:**
1. Go to Render dashboard → Your service → **Logs**
2. Look for:
   - `✓ Model loaded successfully: XGBoost`
   - `Model files not found` error
   - Training completion messages

**Verify model files exist:**
In your build command output, you should see:
```
Saving best model (XGBoost)...
Model saved to: best_model.pkl
Metadata saved to: model_metadata.pkl
```

### Build Succeeds but Model Not Loading at Runtime

**Possible causes:**
1. Model files saved in wrong directory
2. Working directory different at runtime vs build

**Fix:**
Make sure `app.py` uses absolute paths (already done):
```python
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
```

---

## Recommended: Use render.yaml

For easier configuration, create a `render.yaml` file (already created in your repo):

```yaml
services:
  - type: web
    name: breast-cancer-detection
    env: python
    buildCommand: pip install -r requirements.txt && python train_models.py
    startCommand: python app.py
```

Then connect this file in Render:
1. Render Dashboard → New → Blueprint
2. Connect your GitHub repo
3. Render will use `render.yaml` automatically

---

## Quick Fix Checklist

- [ ] Build command includes: `python train_models.py`
- [ ] Build logs show model training completed
- [ ] Build logs show model files saved
- [ ] Health check shows `model_loaded: true`
- [ ] Test prediction works

---

## Example Build Log (Success)

```
Building...
Step 1/3: Installing dependencies
...
✓ pandas installed
✓ scikit-learn installed
...

Step 2/3: Training models
Loading and preprocessing data...
Training set size: 455
...
Best Model: XGBoost (F1-Score: 0.9630)
Saving best model (XGBoost)...
Model saved to: best_model.pkl
Metadata saved to: model_metadata.pkl

Step 3/3: Starting server
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:10000
```

---

## Still Having Issues?

1. Check Render build logs for errors
2. Verify Python version (3.10+ required)
3. Ensure all dependencies install correctly
4. Check if training completes successfully
5. Verify model files are in the correct directory

Good luck! 🚀

