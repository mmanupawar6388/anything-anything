#!/bin/bash
# Flask Deployment Script for Production

echo "🚀 Starting MNIST GAN Flask Deployment..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if model checkpoint exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "⚠️  Warning: No trained model found at checkpoints/best_model.pth"
    echo "   The app will start with an untrained model."
    echo "   Please train the model first using: python scripts/train.py"
fi

# Create necessary directories
mkdir -p checkpoints logs outputs templates static

# Start Flask app with Gunicorn (production)
echo "🌐 Starting Flask app with Gunicorn..."
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --threads 2 \
         --timeout 120 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         wsgi:app

# Alternative: Start with Flask development server
# echo "🌐 Starting Flask development server..."
# python app.py
