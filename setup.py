#!/usr/bin/env python3
"""
Setup script for DocLayout-YOLO Layout Detection Pipeline.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    return True


def install_doclayout_yolo():
    """Install DocLayout-YOLO package."""
    print("Installing DocLayout-YOLO...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "doclayout-yolo"])
        print("✅ DocLayout-YOLO installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Failed to install DocLayout-YOLO, will use ultralytics as fallback")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("✅ Ultralytics YOLO installed as fallback")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install any YOLO package: {e}")
            return False


def download_model():
    """Download the DocLayout-YOLO model."""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "doclayout_yolo_ft.pt"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return True
    
    model_url = "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt"
    
    print(f"Downloading model from: {model_url}")
    print(f"Saving to: {model_path}")
    print("This may take a few minutes...")
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("✅ Model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Please download manually from:")
        print(model_url)
        return False


def create_sample_directories():
    """Create sample input/output directories."""
    directories = ["input", "output", "models"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def update_config():
    """Update config.yaml with correct model path."""
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        print("⚠️  config.yaml not found")
        return
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update model path
    updated_content = content.replace(
        'model_path: "models/doclayout_yolo_ft.pt"',
        'model_path: "models/doclayout_yolo_ft.pt"'
    )
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated config.yaml")


def check_installation():
    """Check if installation was successful."""
    print("\nChecking installation...")
    
    # Check Python packages
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
    
    try:
        import fitz
        print(f"✅ PyMuPDF: {fitz.version[0]}")
    except ImportError:
        print("❌ PyMuPDF not found")
    
    # Check YOLO packages
    try:
        import doclayout_yolo
        print("✅ DocLayout-YOLO available")
    except ImportError:
        try:
            import ultralytics
            print("✅ Ultralytics YOLO available (fallback)")
        except ImportError:
            print("❌ No YOLO package found")
    
    # Check model file
    model_path = Path("models/doclayout_yolo_ft.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file: {model_path} ({size_mb:.1f} MB)")
    else:
        print("❌ Model file not found")
    
    # Check main script
    if Path("layout_detector.py").exists():
        print("✅ Main script: layout_detector.py")
    else:
        print("❌ Main script not found")


def main():
    """Main setup function."""
    print("DocLayout-YOLO Layout Detection Pipeline Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("layout_detector.py").exists():
        print("❌ Please run this script from the layout_detection_pipeline directory")
        return 1
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Install YOLO package
    if not install_doclayout_yolo():
        success = False
    
    # Create directories
    create_sample_directories()
    
    # Download model
    if not download_model():
        success = False
    
    # Update config
    update_config()
    
    # Check installation
    check_installation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your input images/PDFs in the 'input' directory")
        print("2. Run: python run_detection.py")
        print("3. Check results in the 'output' directory")
        print("\nOr try the examples:")
        print("python example.py")
    else:
        print("⚠️  Setup completed with some issues")
        print("Please check the error messages above and resolve them manually")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
