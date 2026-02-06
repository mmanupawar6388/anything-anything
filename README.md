# MNIST GAN - Production Ready Deep Learning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Generative Adversarial Network (GAN) implementation for generating MNIST handwritten digits.

## 🌟 Features

- **State-of-the-art Architecture**: Deep Convolutional GAN (DCGAN) with best practices
- **Comprehensive Training**: Progressive training with learning rate scheduling
- **Monitoring & Logging**: TensorBoard integration, checkpoint management
- **Evaluation Metrics**: FID Score, Inception Score, visual quality assessment
- **Production Ready**: REST API for inference, Docker support
- **Reproducible**: Seed management, configuration files
- **Well Documented**: Type hints, docstrings, comprehensive comments

## 📁 Project Structure

```
mnist-gan/
├── config/
│   ├── default.yaml          # Default configuration
│   └── production.yaml        # Production settings
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py       # Generator architecture
│   │   └── discriminator.py   # Discriminator architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   └── losses.py          # Loss functions
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading utilities
│   │   ├── visualization.py   # Visualization tools
│   │   └── checkpoint.py      # Checkpoint management
│   └── inference/
│       ├── __init__.py
│       └── api.py             # REST API for inference
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── generate.py            # Generation script
├── tests/
│   ├── test_models.py
│   └── test_training.py
├── checkpoints/               # Model checkpoints
├── logs/                      # TensorBoard logs
├── outputs/                   # Generated images
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mnist-gan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom config
python scripts/train.py --config config/production.yaml

# Resume from checkpoint
python scripts/train.py --resume checkpoints/latest.pth
```

### Generate Images

```bash
# Generate 100 images
python scripts/generate.py --num_images 100 --output outputs/

# Generate with specific checkpoint
python scripts/generate.py --checkpoint checkpoints/best_model.pth
```

### Evaluation

```bash
# Evaluate model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

### API Server

```bash
# Start REST API server
python -m uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

# Generate image via API
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"num_images": 5}'
```

### Docker Deployment

```bash
# Build Docker image
docker build -t mnist-gan:latest .

# Run container
docker run -p 8000:8000 mnist-gan:latest

# Using Docker Compose
docker-compose up
```

## 📊 Model Architecture

### Generator
- Input: 100-dimensional latent vector
- Architecture: Transposed Convolutions with BatchNorm and ReLU
- Output: 28x28 grayscale image

### Discriminator
- Input: 28x28 grayscale image
- Architecture: Convolutional layers with LeakyReLU and Dropout
- Output: Real/Fake probability

## 🎯 Performance Metrics

- **FID Score**: Measures quality and diversity of generated images
- **Inception Score**: Evaluates image quality and variety
- **Visual Quality**: Grid visualization of generated samples
- **Training Stability**: Loss curves and gradient monitoring

## 🔧 Configuration

Edit `config/default.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Logging and checkpoint intervals

## 📈 Monitoring

TensorBoard logs are saved in `logs/` directory:

```bash
tensorboard --logdir logs/
```

View at: http://localhost:6006

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py -v
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- MNIST Dataset: Yann LeCun et al.
- DCGAN Paper: Radford et al., 2015
- PyTorch Team for the excellent framework
