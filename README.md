
# Eye-Tracking Keyboard

This project lets you control a virtual keyboard using only your eyes. It uses MediaPipe and OpenCV to track your gaze and simulate key presses through dwell-based selection.

---

## Features

- Real-time gaze tracking using iris landmarks
- 5-point calibration for accurate screen mapping
- Dwell-based key selection (1.2 seconds)
- On-screen QWERTY keyboard with `SPACE`, `DEL`, and `CLEAR`
- One-step installer for all dependencies

---

## Requirements

- Python 3.12.3
- Webcam
- Virtual environment (recommended)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/eye-tracking-keyboard.git
cd eye-tracking-keyboard
```

### 2. Create and activate a virtual environment

```bash
# Create the environment
python3.12 -m venv keyboard_env

# Activate the environment (Windows)
keyboard_env\Scripts\activate

# Activate the environment (macOS/Linux)
source keyboard_env/bin/activate
```

### 3. Install dependencies

```bash
python install_eye_tracking.py
```

This will install:

- `opencv-python>=4.5.0`
- `mediapipe>=0.8.0`
- `numpy>=1.19.0`

---

## Running the Application

```bash
python eye_tracking_keyboard.py
```

---

## Usage

### Calibration (first-time use)

- Look at the red dot on the screen.
- Press `SPACE` when you're looking directly at it.
- Repeat for all 5 calibration points.

### Typing

- Stare at a key on the keyboard to select it.
- Dwell time is approximately 1.2 seconds.
- Available keys:
  - ALL Alphabet letters 
  - `SPACE` — insert a space
  - `DEL` — delete the last character
  - `CLEAR` — clear all typed text

### Controls

- Press `R` to restart calibration
- Press `Q` to quit the application

---

## Project Structure

```text
eye-tracking-keyboard/
├── eye_tracking_keyboard.py     # Main eye tracking application
├── install_eye_tracking.py      # Dependency installer script
├── README.md                    # Project documentation
```

---

## Tips

- Use bright, even lighting for best tracking accuracy.
- Keep your face centered and still in the webcam view.
- Avoid background clutter and movement.
- Works best against a plain wall or neutral background.

---

