"""
REST API for MNIST GAN inference.
Provides endpoints for generating images and model information.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import io
import base64
from PIL import Image
import numpy as np
from pathlib import Path
import yaml

from ..models import Generator
from ..utils.checkpoint import CheckpointManager


# API Models
class GenerateRequest(BaseModel):
    """Request model for image generation."""
    num_images: int = Field(default=1, ge=1, le=100, description="Number of images to generate")
    latent_vectors: Optional[List[List[float]]] = Field(default=None, description="Custom latent vectors")
    return_format: str = Field(default='base64', description="Return format: 'base64' or 'bytes'")


class GenerateResponse(BaseModel):
    """Response model for image generation."""
    images: List[str]
    num_images: int
    latent_dim: int


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    latent_dim: int
    num_parameters: int
    checkpoint_epoch: int
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="MNIST GAN API",
    description="REST API for generating MNIST-like handwritten digits using GAN",
    version="1.0.0"
)


# Global variables for model
generator: Optional[Generator] = None
device: str = 'cpu'
config: dict = {}
checkpoint_info: dict = {}


def load_model(
    checkpoint_path: str = './checkpoints/best_model.pth',
    config_path: str = './config/default.yaml',
    device_name: str = 'cuda'
):
    """Load the generator model from checkpoint."""
    global generator, device, config, checkpoint_info
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu'
    
    # Create generator
    generator = Generator(
        latent_dim=config['model']['latent_dim'],
        channels=config['model']['generator']['channels']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    # Store checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0)
    }
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Epoch: {checkpoint_info['epoch']}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        # Try to load best model
        if not checkpoint_path.exists():
            checkpoint_path = Path('./checkpoints/final_model.pth')
        if not checkpoint_path.exists():
            checkpoint_path = Path('./checkpoints/interrupted.pth')
        
        if checkpoint_path.exists():
            load_model(str(checkpoint_path))
        else:
            print("⚠ No checkpoint found. Model will be loaded on first request.")
    except Exception as e:
        print(f"⚠ Failed to load model on startup: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MNIST GAN API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate",
            "info": "/info",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": generator is not None
    }


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    num_params = sum(p.numel() for p in generator.parameters())
    
    return ModelInfo(
        model_name="MNIST GAN",
        latent_dim=config['model']['latent_dim'],
        num_parameters=num_params,
        checkpoint_epoch=checkpoint_info.get('epoch', 0),
        device=device
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_images(request: GenerateRequest):
    """
    Generate MNIST-like images.
    
    Args:
        request: Generation request with parameters
    
    Returns:
        Generated images in base64 format
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate latent vectors
        if request.latent_vectors is not None:
            z = torch.tensor(request.latent_vectors, dtype=torch.float32, device=device)
            num_images = len(request.latent_vectors)
        else:
            num_images = request.num_images
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
            images_base64.append(img_base64)
        
        return GenerateResponse(
            images=images_base64,
            num_images=num_images,
            latent_dim=config['model']['latent_dim']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/image")
async def generate_single_image():
    """
    Generate a single image and return as PNG.
    
    Returns:
        PNG image
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
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
        
        return StreamingResponse(buffer, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Load model
    load_model()
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
