# Flask Deployment Configuration

# Development
DEBUG = False
TESTING = False

# Server
HOST = '0.0.0.0'
PORT = 5000

# Model
MODEL_CHECKPOINT = 'checkpoints/best_model.pth'
CONFIG_PATH = 'config/default.yaml'

# Security
SECRET_KEY = 'change-this-in-production'  # Change this!
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size

# CORS
CORS_ORIGINS = '*'  # Restrict in production

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Generation limits
MAX_IMAGES_PER_REQUEST = 100
MAX_INTERPOLATION_STEPS = 50
