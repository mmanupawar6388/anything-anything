# MNIST GAN Project - Complete Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [API Reference](#api-reference)
6. [Training Details](#training-details)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

This is a production-ready implementation of a Generative Adversarial Network (GAN) for generating MNIST handwritten digits. The project follows industry best practices and is ready for deployment.

### Key Features
- ✅ DCGAN architecture with proven performance
- ✅ Comprehensive training pipeline with monitoring
- ✅ Multiple evaluation metrics (FID, IS, visual quality)
- ✅ REST API for inference
- ✅ Docker support for easy deployment
- ✅ Extensive documentation and tests
- ✅ TensorBoard integration
- ✅ Checkpoint management with auto-cleanup

### Model Statistics
- **Generator Parameters**: ~1.9M
- **Discriminator Parameters**: ~660K
- **Total Parameters**: ~2.6M
- **Input**: 100-dimensional latent vector
- **Output**: 28x28 grayscale image

## 🏗️ Architecture

### Generator Network
```
Input (100) 
  → Linear + BatchNorm + ReLU → Reshape (256, 7, 7)
  → ConvTranspose2d (256→128) + BatchNorm + ReLU → (128, 14, 14)
  → ConvTranspose2d (128→64) + BatchNorm + ReLU → (64, 28, 28)
  → Conv2d (64→1) + Tanh → (1, 28, 28)
```

### Discriminator Network
```
Input (1, 28, 28)
  → Conv2d (1→64) + LeakyReLU + Dropout → (64, 14, 14)
  → Conv2d (64→128) + BatchNorm + LeakyReLU + Dropout → (128, 7, 7)
  → Conv2d (128→256) + BatchNorm + LeakyReLU + Dropout → (256, 3, 3)
  → Flatten + Linear + Sigmoid → (1)
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU training)

### Quick Install
```bash
# Clone repository
git clone <your-repo-url>
cd mnist-gan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
python src/models/generator.py
python src/models/discriminator.py
```

## 🚀 Usage Guide

### Interactive Quick Start
```bash
python quickstart.py
```

This provides an interactive menu for all common tasks.

### Manual Training

#### Quick Training (10 epochs, for testing)
```bash
python scripts/train.py --epochs 10
```

#### Default Training (100 epochs)
```bash
python scripts/train.py
```

#### Production Training (200 epochs)
```bash
python scripts/train.py --config config/production.yaml
```

#### Resume Training
```bash
python scripts/train.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Generate Images

#### Generate 64 images
```bash
python scripts/generate.py --num_images 64 --output outputs/generated.png
```

#### Generate with specific checkpoint
```bash
python scripts/generate.py --checkpoint checkpoints/best_model.pth --num_images 100
```

#### Generate latent space interpolation
```bash
python scripts/generate.py --interpolate --num_images 100
```

### Evaluate Model

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --num_samples 10000
```

### Start API Server

```bash
python -m uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

Access API documentation at: http://localhost:8000/docs

## 🔌 API Reference

### Endpoints

#### GET /
Returns API information and available endpoints.

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### GET /info
Get model information.

**Response:**
```json
{
  "model_name": "MNIST GAN",
  "latent_dim": 100,
  "num_parameters": 1948353,
  "checkpoint_epoch": 100,
  "device": "cuda"
}
```

#### POST /generate
Generate multiple images.

**Request:**
```json
{
  "num_images": 10,
  "return_format": "base64"
}
```

**Response:**
```json
{
  "images": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
  "num_images": 10,
  "latent_dim": 100
}
```

#### POST /generate/image
Generate a single image and return as PNG.

**Response:** PNG image

### Example API Usage

#### Python
```python
import requests
import base64
from PIL import Image
import io

# Generate images
response = requests.post(
    "http://localhost:8000/generate",
    json={"num_images": 5}
)

data = response.json()

# Decode and save first image
img_data = base64.b64decode(data['images'][0])
img = Image.open(io.BytesIO(img_data))
img.save('generated.png')
```

#### cURL
```bash
# Get model info
curl http://localhost:8000/info

# Generate single image
curl -X POST http://localhost:8000/generate/image --output generated.png

# Generate multiple images
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_images": 10}'
```

## 🎓 Training Details

### Hyperparameters (Default)
- **Batch Size**: 128
- **Learning Rate**: 0.0002 (both G and D)
- **Beta1**: 0.5
- **Beta2**: 0.999
- **Epochs**: 100
- **Label Smoothing**: 0.1
- **Noise Std**: 0.1

### Training Techniques
1. **Label Smoothing**: Reduces discriminator overconfidence
2. **Noise Injection**: Adds stability to training
3. **Gradient Clipping**: Prevents exploding gradients
4. **Learning Rate Scheduling**: Adaptive learning rates
5. **Batch Normalization**: Stabilizes training
6. **Dropout**: Prevents overfitting in discriminator

### Monitoring
- **TensorBoard**: Real-time training visualization
- **Checkpoints**: Saved every 5 epochs
- **Sample Images**: Generated every 500 batches
- **Metrics Logging**: Loss, accuracy, learning rate

### View Training Progress
```bash
tensorboard --logdir logs/
```

Access at: http://localhost:6006

## 📊 Evaluation Metrics

### FID Score (Fréchet Inception Distance)
- Measures quality and diversity of generated images
- **Lower is better**
- Compares feature distributions of real and fake images
- Good score: < 50

### Inception Score
- Measures quality and variety
- **Higher is better**
- Evaluates how well images match expected classes
- Good score: > 5

### Visual Quality Metrics
- **Mean/Std**: Distribution statistics
- **Contrast**: Dynamic range
- **Sharpness**: Edge definition

## 🐳 Deployment

### Docker Deployment

#### Build Image
```bash
docker build -t mnist-gan:latest .
```

#### Run Container
```bash
docker run -p 8000:8000 mnist-gan:latest
```

#### Using Docker Compose
```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down
```

### Services
- **API Server**: Port 8000
- **TensorBoard**: Port 6006

### Production Considerations
1. **GPU Support**: Use CUDA-enabled Docker image
2. **Model Checkpoints**: Mount volume for checkpoints
3. **Scaling**: Use multiple workers with uvicorn
4. **Monitoring**: Integrate with logging services
5. **Security**: Add authentication for production

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (OOM)
**Solution**: Reduce batch size in config
```yaml
training:
  batch_size: 64  # Reduce from 128
```

#### Training Instability
**Solutions**:
1. Reduce learning rate
2. Increase label smoothing
3. Add more noise to discriminator inputs

#### Poor Image Quality
**Solutions**:
1. Train for more epochs
2. Adjust architecture (more channels)
3. Try different loss functions (WGAN, Hinge)

#### API Not Loading Model
**Solution**: Ensure checkpoint exists
```bash
ls checkpoints/best_model.pth
```

### Getting Help
1. Check logs in `logs/` directory
2. Review TensorBoard for training issues
3. Run tests: `pytest tests/ -v`
4. Check GitHub issues

## 📝 Configuration Reference

### Model Configuration
```yaml
model:
  latent_dim: 100  # Size of latent vector
  generator:
    channels: [256, 128, 64, 1]  # Channel progression
  discriminator:
    channels: [1, 64, 128, 256]
    dropout: 0.3  # Dropout probability
```

### Training Configuration
```yaml
training:
  batch_size: 128
  num_epochs: 100
  learning_rate:
    generator: 0.0002
    discriminator: 0.0002
  label_smoothing: 0.1
  noise_std: 0.1
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Tests
```bash
pytest tests/test_models.py -v
pytest tests/test_training.py -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Benchmarks

### Training Time (GPU - RTX 3080)
- 10 epochs: ~5 minutes
- 100 epochs: ~45 minutes
- 200 epochs: ~90 minutes

### Training Time (CPU)
- 10 epochs: ~30 minutes
- 100 epochs: ~5 hours

### Inference Speed
- GPU: ~1000 images/second
- CPU: ~50 images/second

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- MNIST Dataset: Yann LeCun et al.
- DCGAN Paper: Radford et al., 2015
- PyTorch Team

---

**Built with ❤️ for production deployment**
