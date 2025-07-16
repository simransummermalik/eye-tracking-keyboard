import subprocess
import sys

def install_packages():
    """Install required packages for eye tracking"""
    packages = [
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0", 
        "numpy>=1.19.0"
    ]
    
    print("🔧 Installing Eye Tracking Dependencies...")
    print("=" * 50)
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f" {package} installed successfully!")
        except subprocess.CalledProcessError:
            print(f" Failed to install {package}")
            return False
    
    print("\n All packages installed successfully!")
    print("\n Next steps:")
    print("1. Run: python eye_tracking_keyboard.py")
    print("2. Make sure you have good lighting")
    print("3. Position your face clearly in the camera")
    print("4. Look at keys and dwell (stare) for 1.5 seconds to click")
    
    return True

if __name__ == "__main__":
    install_packages()
