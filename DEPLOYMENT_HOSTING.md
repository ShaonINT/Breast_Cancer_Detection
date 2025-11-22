# Deployment Guide for Traditional Web Hosting

This guide covers deploying the Breast Cancer Detection API on traditional web hosting providers (cPanel, shared hosting, VPS with unlimited domains).

---

## Prerequisites

1. **Check hosting requirements:**
   - Python 3.8+ support
   - SSH access (recommended) or cPanel File Manager
   - Ability to install Python packages
   - Ability to run Python scripts
   - Port access (usually port 80/443, some allow custom ports)

2. **Verify Python version:**
   ```bash
   python --version  # or python3 --version
   ```

---

## Method 1: SSH Access (Recommended)

If your hosting provider gives you SSH access, this is the easiest method.

### Step 1: Upload Files

**Option A: Using Git (if available)**
```bash
# SSH into your server
ssh username@your-domain.com

# Navigate to your domain directory
cd public_html  # or www, or domains/yourdomain.com/public_html

# Clone your repository
git clone https://github.com/ShaonINT/Breast_Cancer_Detection.git .

# Or pull if already cloned
git pull origin main
```

**Option B: Using SFTP/FTP**
1. Use FileZilla, WinSCP, or any FTP client
2. Upload all project files to your domain directory
3. Upload to: `/public_html/` or `/domains/yourdomain.com/public_html/`

### Step 2: Create Virtual Environment

```bash
# SSH into your server
ssh username@your-domain.com

# Navigate to project directory
cd public_html  # or your domain directory

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Train the model
python train_models.py
```

### Step 4: Test Locally on Server

```bash
# Test the application
python app.py

# Should start on port 8000
# Test: curl http://localhost:8000/health
```

### Step 5: Set Up Process Manager

**Using nohup (simple):**
```bash
# Start server in background
nohup python app.py > app.log 2>&1 &

# Check if running
ps aux | grep python

# View logs
tail -f app.log
```

**Using Supervisor (better for production):**
```bash
# Install supervisor
sudo apt-get install supervisor  # Ubuntu/Debian
# or
sudo yum install supervisor  # CentOS/RHEL

# Create config file
sudo nano /etc/supervisor/conf.d/breast-cancer-api.conf
```

Add this configuration:
```ini
[program:breast-cancer-api]
command=/path/to/venv/bin/python /path/to/public_html/app.py
directory=/path/to/public_html
user=your-username
autostart=true
autorestart=true
stderr_logfile=/var/log/breast-cancer-api.err.log
stdout_logfile=/var/log/breast-cancer-api.out.log
environment=PATH="/path/to/venv/bin"
```

Then:
```bash
# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start breast-cancer-api

# Check status
sudo supervisorctl status
```

### Step 6: Set Up Reverse Proxy (nginx/Apache)

Your hosting provider likely has nginx or Apache configured. You need to set up a reverse proxy.

#### For nginx:

Edit nginx configuration (usually in `/etc/nginx/sites-available/your-domain`):

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Reload nginx:
```bash
sudo nginx -t  # Test configuration
sudo systemctl reload nginx  # Reload
```

#### For Apache:

Edit Apache configuration or `.htaccess`:

**In .htaccess file:**
```apache
<IfModule mod_proxy.c>
    ProxyPass / http://127.0.0.1:8000/
    ProxyPassReverse / http://127.0.0.1:8000/
    ProxyPreserveHost On
</IfModule>
```

**Or in VirtualHost configuration:**
```apache
<VirtualHost *:80>
    ServerName your-domain.com
    ServerAlias www.your-domain.com
    
    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8000/
    ProxyPassReverse / http://127.0.0.1:8000/
</VirtualHost>
```

Enable modules and restart:
```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo systemctl restart apache2
```

---

## Method 2: cPanel Deployment

If you only have cPanel access (no SSH), follow these steps:

### Step 1: Upload Files via File Manager

1. Log into cPanel
2. Go to **File Manager**
3. Navigate to `public_html` or your domain folder
4. Upload all project files (zip and extract, or upload individually)

### Step 2: Check Python Support

1. In cPanel, look for **Python** or **Setup Python App**
2. If available, go to **Setup Python App**

### Step 3: Create Python App

1. Click **Create Application**
2. Select Python version: **3.10** or latest available
3. Choose your domain/subdomain
4. Set application root: `/public_html` or your domain folder
5. Click **Create**

### Step 4: Install Dependencies

1. In **Python App** settings, click **Edit**
2. Go to **Virtual Environment** or **Modules**
3. Install packages manually or upload `requirements.txt` and install:

```bash
# In the Python app terminal/SSH (if available)
pip install -r requirements.txt
```

**Or manually install each package:**
```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
catboost==1.2.2
lightgbm==4.1.0
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
joblib==1.3.2
python-multipart==0.0.6
```

### Step 5: Update app.py for cPanel

Create a modified version of `app.py` for cPanel (or update existing):

```python
# At the end of app.py, change the main block:
if __name__ == "__main__":
    import uvicorn
    import os
    
    # For cPanel, use the port assigned by hosting or WSGI
    port = int(os.environ.get("PORT", 8000))
    
    # cPanel often uses specific host configurations
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
```

### Step 6: Create WSGI Entry Point (if required)

Some hosts require a WSGI entry point. Create `passenger_wsgi.py`:

```python
# passenger_wsgi.py
import sys
import os

# Add your project directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import the app
from app import app

# For Passenger/Phusion Passenger
if __name__ == "__main__":
    app.run()
```

### Step 7: Configure Start Script

In cPanel **Python App**, set:
- **Startup File**: `app.py` or `passenger_wsgi.py`
- **App URL**: Your domain or subdomain

### Step 8: Train Model

If you have SSH access through cPanel:
1. Open **Terminal** in cPanel
2. Navigate to your project directory
3. Activate Python app environment (path shown in Python App settings)
4. Run: `python train_models.py`

**Or use Python App's Run Script feature:**
- Go to Python App → Run Script
- Select `train_models.py`
- Click Run

---

## Method 3: Using Subdomain or Addon Domain

If you have unlimited domains, create a dedicated subdomain:

### Steps:

1. **Create subdomain** (e.g., `api.yourdomain.com` or `cancer.yourdomain.com`):
   - cPanel → Subdomains → Create
   - Or use Addon Domain

2. **Upload files** to subdomain directory:
   - Usually: `/public_html/subdomain_name` or `/domains/subdomain_name/public_html`

3. **Follow Method 1 or 2** above for the subdomain

---

## Method 4: Using .htaccess for Apache (Simplified)

If you can't use reverse proxy, create a simple PHP proxy:

**Create `index.php` in your domain root:**

```php
<?php
// Simple proxy to FastAPI backend
$url = "http://127.0.0.1:8000" . $_SERVER['REQUEST_URI'];

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);

// Forward headers
$headers = array();
foreach (getallheaders() as $name => $value) {
    $headers[] = "$name: $value";
}
curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

// Forward method
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $_SERVER['REQUEST_METHOD']);

// Forward POST data
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    curl_setopt($ch, CURLOPT_POSTFIELDS, file_get_contents('php://input'));
}

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

// Set headers
http_response_code($httpCode);
header('Content-Type: application/json');

echo $response;
?>
```

**Start the Python app in background** (via cron or nohup).

---

## Important Configuration Files

### 1. Create `.htaccess` for Apache (if needed)

```apache
# .htaccess for redirecting to Python app
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ http://127.0.0.1:8000/$1 [P,L]
```

### 2. Update app.py for hosting

Ensure `app.py` uses environment variables:

```python
# Already done, but verify:
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 3. File Permissions

Set correct permissions:
```bash
chmod 755 app.py
chmod 755 train_models.py
chmod 755 deploy.sh
chmod 644 *.pkl  # Model files
chmod 644 requirements.txt
```

---

## Troubleshooting

### Issue: Port 8000 not accessible

**Solution:**
- Use the port provided by your hosting (check environment variables)
- Some hosts use port 5000, 8080, or random ports
- Check: `echo $PORT` in SSH

### Issue: Python packages won't install

**Solution:**
- Use virtual environment
- Install with `--user` flag: `pip install --user -r requirements.txt`
- Check Python version: `python3 --version`

### Issue: Model files not found

**Solution:**
- Train model on server: `python train_models.py`
- Or upload pre-trained model files via FTP
- Check file paths in `app.py`

### Issue: App stops after SSH disconnect

**Solution:**
- Use `nohup` or `screen` or `tmux`
- Use process manager like Supervisor
- Use cPanel Python App (keeps running)

### Issue: Cannot access from browser

**Solution:**
- Check firewall settings
- Verify reverse proxy configuration
- Check if app is running: `ps aux | grep python`
- Check logs for errors

---

## Quick Setup Script for Hosting

Create `setup_hosting.sh`:

```bash
#!/bin/bash

echo "Setting up Breast Cancer Detection on hosting..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model
echo "Training model..."
python train_models.py

# Set permissions
chmod 755 app.py train_models.py

echo "Setup complete!"
echo "To start: source venv/bin/activate && python app.py"
```

Make executable:
```bash
chmod +x setup_hosting.sh
./setup_hosting.sh
```

---

## Testing Your Deployment

1. **Health Check:**
   ```
   http://your-domain.com/health
   ```

2. **API Docs:**
   ```
   http://your-domain.com/docs
   ```

3. **Web Interface:**
   ```
   http://your-domain.com
   ```

4. **Test Prediction:**
   ```bash
   curl -X POST "http://your-domain.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"radius_mean": 17.99, ...}'
   ```

---

## Support from Hosting Provider

If you encounter issues, contact your hosting provider and ask:

1. **Python version** available
2. **SSH access** available
3. **Process management** (Supervisor, systemd, etc.)
4. **Reverse proxy** setup (nginx/Apache configuration)
5. **Port restrictions**
6. **Cron jobs** for auto-restart

---

## Recommended Approach

**Best for traditional hosting:**
1. Use SSH access if available
2. Set up virtual environment
3. Use Supervisor or systemd for process management
4. Configure nginx/Apache reverse proxy
5. Train model on server or upload pre-trained files

This ensures your app runs reliably and automatically restarts if it crashes.

---

## Security Notes

1. **Don't expose sensitive data** in logs
2. **Use HTTPS** if available (SSL certificate)
3. **Restrict CORS** to your domain in production:
   ```python
   allow_origins=["https://your-domain.com"]
   ```
4. **Set proper file permissions**
5. **Keep dependencies updated**

Good luck with your deployment! 🚀

