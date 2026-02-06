# 🌐 Flask Deployment Guide - MNIST GAN

## 🚀 Quick Start

### Option 1: Run Locally (Development)
```bash
# Install Flask dependencies
pip install flask flask-cors gunicorn

# Start Flask development server
python app.py
```

Access at: **http://localhost:5000**

### Option 2: Run with Deployment Script (Windows)
```bash
# Double-click or run:
deploy_flask.bat
```

### Option 3: Run with Deployment Script (Linux/Mac)
```bash
# Make executable
chmod +x deploy_flask.sh

# Run
./deploy_flask.sh
```

### Option 4: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.flask.yml up

# Or build manually
docker build -f Dockerfile.flask -t mnist-gan-flask .
docker run -p 5000:5000 mnist-gan-flask
```

---

## 📋 Features

### ✨ Web Interface
- **Beautiful UI** with gradient design and animations
- **Real-time generation** with loading indicators
- **Image gallery** with hover effects
- **Click to download** individual images
- **Model information** dashboard
- **Responsive design** for all devices

### 🔌 API Endpoints

#### GET `/`
Main web interface

#### GET `/health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": "2026-02-05T21:20:00"
}
```

#### GET `/api/info`
Get model information
```json
{
  "model_name": "MNIST GAN",
  "architecture": "DCGAN",
  "latent_dim": 100,
  "num_parameters": 1948353,
  "device": "cpu",
  "status": "ready"
}
```

#### POST `/api/generate`
Generate multiple images

**Request:**
```json
{
  "num_images": 16,
  "seed": 42
}
```

**Response:**
```json
{
  "success": true,
  "images": ["data:image/png;base64,..."],
  "num_images": 16,
  "latent_dim": 100,
  "seed": 42
}
```

#### GET/POST `/api/generate/single`
Generate a single PNG image

**Query params:** `?seed=42`

**Response:** PNG image

#### POST `/api/interpolate`
Generate latent space interpolation

**Request:**
```json
{
  "num_steps": 20,
  "seed1": 42,
  "seed2": 123
}
```

**Response:**
```json
{
  "success": true,
  "images": ["data:image/png;base64,..."],
  "num_steps": 20
}
```

---

## 🎨 Using the Web Interface

### Generate Images
1. Open http://localhost:5000
2. Set number of images (1-100)
3. Optionally set a random seed
4. Click "Generate Images"
5. Click any image to download

### Generate Interpolation
1. Click "Interpolate" button
2. Watch smooth transitions between random digits
3. Download your favorites

### Keyboard Shortcuts
- **Enter**: Generate images
- **Ctrl+I**: Generate interpolation
- **Ctrl+Shift+C**: Clear gallery

---

## 🐍 Python API Client Example

```python
import requests
import base64
from PIL import Image
import io

# Base URL
BASE_URL = "http://localhost:5000"

# Generate images
response = requests.post(
    f"{BASE_URL}/api/generate",
    json={"num_images": 10, "seed": 42}
)

data = response.json()

# Save images
for i, img_b64 in enumerate(data['images']):
    # Remove data URL prefix
    img_data = img_b64.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    img.save(f'digit_{i}.png')

print(f"Saved {len(data['images'])} images!")
```

---

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_APP=app.py
export FLASK_ENV=production  # or development
export MODEL_CHECKPOINT=checkpoints/best_model.pth
```

### config.py Settings
```python
# Server
HOST = '0.0.0.0'
PORT = 5000

# Model
MODEL_CHECKPOINT = 'checkpoints/best_model.pth'

# Generation limits
MAX_IMAGES_PER_REQUEST = 100
MAX_INTERPOLATION_STEPS = 50
```

---

## 🚀 Production Deployment

### With Gunicorn (Recommended)
```bash
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --threads 2 \
         --timeout 120 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         wsgi:app
```

### With Docker
```bash
# Build
docker build -f Dockerfile.flask -t mnist-gan-flask .

# Run
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  --name mnist-gan-flask \
  mnist-gan-flask
```

### With Docker Compose
```bash
docker-compose -f docker-compose.flask.yml up -d
```

---

## 🌐 Deployment Platforms

### Heroku
```bash
# Create Procfile
echo "web: gunicorn wsgi:app" > Procfile

# Deploy
heroku create mnist-gan-app
git push heroku main
```

### AWS EC2
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip nginx

# Install app
pip3 install -r requirements.txt

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/mnist-gan-flask
gcloud run deploy --image gcr.io/PROJECT_ID/mnist-gan-flask --platform managed
```

### DigitalOcean App Platform
1. Connect GitHub repository
2. Select Dockerfile.flask
3. Set port to 5000
4. Deploy!

---

## 📊 Performance Tuning

### Gunicorn Workers
```bash
# Formula: (2 x CPU cores) + 1
# For 4 cores:
gunicorn --workers 9 --threads 2 wsgi:app
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🔒 Security Considerations

### Production Checklist
- [ ] Change SECRET_KEY in config.py
- [ ] Set DEBUG = False
- [ ] Configure CORS properly (restrict origins)
- [ ] Add rate limiting
- [ ] Enable HTTPS/SSL
- [ ] Add authentication if needed
- [ ] Set up firewall rules
- [ ] Configure logging

### Rate Limiting Example
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)

@app.route('/api/generate')
@limiter.limit("10 per minute")
def generate_images():
    # ...
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

### Model Not Loading
```bash
# Check checkpoint exists
ls checkpoints/best_model.pth

# Train model if needed
python scripts/train.py --epochs 100
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📈 Monitoring

### Access Logs
```bash
tail -f logs/access.log
```

### Error Logs
```bash
tail -f logs/error.log
```

### Health Check
```bash
curl http://localhost:5000/health
```

---

## 🎯 Next Steps

1. ✅ **Train Model**: `python scripts/train.py --epochs 100`
2. ✅ **Test Locally**: `python app.py`
3. ✅ **Deploy**: Choose your platform
4. ✅ **Monitor**: Check logs and health
5. ✅ **Scale**: Add more workers/instances

---

## 📞 API Testing with cURL

```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/api/info

# Generate single image
curl http://localhost:5000/api/generate/single -o digit.png

# Generate multiple images
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"num_images": 5, "seed": 42}'

# Generate interpolation
curl -X POST http://localhost:5000/api/interpolate \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 10}'
```

---

## ✅ Deployment Checklist

- [x] Flask app created (app.py)
- [x] Beautiful web interface (templates/index.html)
- [x] REST API endpoints
- [x] WSGI entry point (wsgi.py)
- [x] Configuration file (config.py)
- [x] Deployment scripts (deploy_flask.bat/sh)
- [x] Dockerfile for Flask
- [x] Docker Compose configuration
- [x] Requirements updated
- [ ] Model trained (run: python scripts/train.py)
- [ ] Production deployment

---

**🎉 Your Flask deployment is ready!**

**Start now:**
```bash
python app.py
```

**Then visit:** http://localhost:5000

---

Built with ❤️ using Flask & PyTorch
