"""
Flask Application for MNIST GAN Deployment.
Production-ready Flask API with web interface.
"""

from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
import torch
import io
import base64
from PIL import Image
import numpy as np
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime

from src.models import Generator

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
generator = None
device = 'cpu'
config = {}
model_info = {}


def load_model(checkpoint_path=None, config_path='config/default.yaml'):
    """Load the generator model from checkpoint."""
    global generator, device, config, model_info
    
    try:
        # Load config
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Create generator
        generator = Generator(
            latent_dim=config['model']['latent_dim'],
            channels=config['model']['generator']['channels']
        )
        
        # Try to find best available checkpoint
        checkpoint_paths = [
            'checkpoints/best_model.pth',
            'checkpoints/final_model.pth',
            'checkpoints/checkpoint_epoch_5.pth',
            'checkpoints/interrupted.pth',
        ]
        
        if checkpoint_path:
            checkpoint_paths.insert(0, checkpoint_path)
        
        loaded = False
        for ckpt_path in checkpoint_paths:
            checkpoint_file = Path(ckpt_path)
            if checkpoint_file.exists():
                logger.info(f"Loading checkpoint from {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device)
                generator.load_state_dict(checkpoint['generator_state_dict'])
                
                model_info = {
                    'epoch': checkpoint.get('epoch', 0),
                    'global_step': checkpoint.get('global_step', 0),
                    'loaded': True,
                    'checkpoint_path': str(ckpt_path)
                }
                loaded = True
                break
        
        if not loaded:
            logger.warning(f"No checkpoint found. Using untrained model.")
            model_info = {
                'epoch': 0,
                'global_step': 0,
                'loaded': False,
                'checkpoint_path': None
            }
        
        generator.to(device)
        generator.eval()
        
        logger.info("✓ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


# Load model on startup
load_model()


@app.route('/')
def index():
    """Render the main web interface."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': generator is not None,
        'device': device,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/info')
def get_info():
    """Get model information."""
    if generator is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    num_params = sum(p.numel() for p in generator.parameters())
    
    return jsonify({
        'model_name': 'MNIST GAN',
        'architecture': 'DCGAN',
        'latent_dim': config['model']['latent_dim'],
        'num_parameters': num_params,
        'device': device,
        'model_info': model_info,
        'status': 'ready'
    })


@app.route('/api/reload', methods=['POST'])
def reload_model_endpoint():
    """Reload the model from the best available checkpoint."""
    try:
        success = load_model()
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully',
                'model_info': model_info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to reload model'
            }), 500
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate_images():
    """
    Generate MNIST-like images.
    
    Request JSON:
    {
        "num_images": 10,
        "seed": 42 (optional),
        "return_format": "base64" or "url"
    }
    """
    if generator is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json() or {}
        num_images = min(int(data.get('num_images', 1)), 100)  # Max 100 images
        seed = data.get('seed', None)
        return_format = data.get('return_format', 'base64')
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate latent vectors
        z = torch.randn(num_images, config['model']['latent_dim'], device=device)
        
        # Generate images
        with torch.no_grad():
            generated = generator(z)
            # Denormalize from [-1, 1] to [0, 1]
            generated = (generated + 1) / 2
            generated = generated.cpu().numpy()
        
        # Convert to base64
        images_base64 = []
        for i in range(num_images):
            img_array = (generated[i, 0] * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images_base64.append(f"data:image/png;base64,{img_base64}")
        
        return jsonify({
            'success': True,
            'images': images_base64,
            'num_images': num_images,
            'latent_dim': config['model']['latent_dim'],
            'seed': seed
        })
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/single', methods=['GET', 'POST'])
def generate_single_image():
    """Generate a single image and return as PNG."""
    if generator is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Get seed from query params or JSON
        if request.method == 'POST':
            data = request.get_json() or {}
            seed = data.get('seed', None)
        else:
            seed = request.args.get('seed', None, type=int)
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image
        z = torch.randn(1, config['model']['latent_dim'], device=device)
        
        with torch.no_grad():
            generated = generator(z)
            generated = (generated + 1) / 2
            img_array = (generated[0, 0].cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array, mode='L')
        
        # Return as PNG
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/interpolate', methods=['POST'])
def interpolate_latent():
    """
    Generate interpolation between two random points in latent space.
    
    Request JSON:
    {
        "num_steps": 10,
        "seed1": 42 (optional),
        "seed2": 123 (optional)
    }
    """
    if generator is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json() or {}
        num_steps = min(int(data.get('num_steps', 10)), 50)
        seed1 = data.get('seed1', None)
        seed2 = data.get('seed2', None)
        
        # Generate two random latent vectors
        if seed1 is not None:
            torch.manual_seed(seed1)
        z1 = torch.randn(1, config['model']['latent_dim'], device=device)
        
        if seed2 is not None:
            torch.manual_seed(seed2)
        z2 = torch.randn(1, config['model']['latent_dim'], device=device)
        
        # Interpolate
        alphas = torch.linspace(0, 1, num_steps, device=device)
        images_base64 = []
        
        with torch.no_grad():
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                generated = generator(z)
                generated = (generated + 1) / 2
                
                img_array = (generated[0, 0].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images_base64.append(f"data:image/png;base64,{img_base64}")
        
        return jsonify({
            'success': True,
            'images': images_base64,
            'num_steps': num_steps
        })
    
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False in production
        threaded=True
    )
