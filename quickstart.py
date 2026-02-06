"""
Quick start script for MNIST GAN project.
This script provides an interactive way to train, evaluate, or generate images.
"""

import subprocess
import sys
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("  MNIST GAN - Production Ready Deep Learning Project")
    print("="*60 + "\n")


def print_menu():
    """Print main menu."""
    print("\nWhat would you like to do?")
    print("1. Train a new model")
    print("2. Resume training from checkpoint")
    print("3. Generate images")
    print("4. Evaluate model")
    print("5. Start API server")
    print("6. Run tests")
    print("7. Install dependencies")
    print("0. Exit")
    print()


def install_dependencies():
    """Install project dependencies."""
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\n✓ Dependencies installed!")


def train_model():
    """Train a new model."""
    print("\nTraining Configuration:")
    print("1. Quick training (10 epochs, for testing)")
    print("2. Default training (100 epochs)")
    print("3. Production training (200 epochs)")
    print("4. Custom configuration")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        cmd = [sys.executable, "scripts/train.py", "--epochs", "10"]
    elif choice == "2":
        cmd = [sys.executable, "scripts/train.py"]
    elif choice == "3":
        cmd = [sys.executable, "scripts/train.py", "--config", "config/production.yaml"]
    elif choice == "4":
        config = input("Enter config path (default: config/default.yaml): ").strip()
        epochs = input("Enter number of epochs (default: 100): ").strip()
        
        cmd = [sys.executable, "scripts/train.py"]
        if config:
            cmd.extend(["--config", config])
        if epochs:
            cmd.extend(["--epochs", epochs])
    else:
        print("Invalid choice!")
        return
    
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def resume_training():
    """Resume training from checkpoint."""
    checkpoint = input("\nEnter checkpoint path (e.g., checkpoints/checkpoint_epoch_50.pth): ").strip()
    
    if not checkpoint:
        print("No checkpoint specified!")
        return
    
    cmd = [sys.executable, "scripts/train.py", "--resume", checkpoint]
    
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def generate_images():
    """Generate images."""
    print("\nGeneration Options:")
    
    checkpoint = input("Checkpoint path (default: checkpoints/best_model.pth): ").strip()
    if not checkpoint:
        checkpoint = "checkpoints/best_model.pth"
    
    num_images = input("Number of images (default: 64): ").strip()
    if not num_images:
        num_images = "64"
    
    output = input("Output path (default: outputs/generated.png): ").strip()
    if not output:
        output = "outputs/generated.png"
    
    interpolate = input("Generate interpolation? (y/n, default: n): ").strip().lower()
    
    cmd = [
        sys.executable, "scripts/generate.py",
        "--checkpoint", checkpoint,
        "--num_images", num_images,
        "--output", output
    ]
    
    if interpolate == 'y':
        cmd.append("--interpolate")
    
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def evaluate_model():
    """Evaluate model."""
    checkpoint = input("\nEnter checkpoint path (e.g., checkpoints/best_model.pth): ").strip()
    
    if not checkpoint:
        print("No checkpoint specified!")
        return
    
    num_samples = input("Number of samples for evaluation (default: 10000): ").strip()
    if not num_samples:
        num_samples = "10000"
    
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", checkpoint,
        "--num_samples", num_samples
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def start_api():
    """Start API server."""
    print("\nStarting API server...")
    print("API will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.inference.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    subprocess.run(cmd)


def run_tests():
    """Run tests."""
    print("\nRunning tests...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def main():
    """Main function."""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("scripts/train.py").exists():
        print("Error: Please run this script from the project root directory!")
        sys.exit(1)
    
    while True:
        print_menu()
        choice = input("Enter your choice (0-7): ").strip()
        
        if choice == "0":
            print("\nGoodbye! 👋\n")
            break
        elif choice == "1":
            train_model()
        elif choice == "2":
            resume_training()
        elif choice == "3":
            generate_images()
        elif choice == "4":
            evaluate_model()
        elif choice == "5":
            start_api()
        elif choice == "6":
            run_tests()
        elif choice == "7":
            install_dependencies()
        else:
            print("\n❌ Invalid choice! Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋\n")
        sys.exit(0)
