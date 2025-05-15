# Eye Cursor Control v1.0.0

**Eye Cursor Control** is a lightweight Python application that uses webcam-based eye and hand tracking to control the mouse pointer. Built with MediaPipe, it enables hands-free control via gaze and gesture-based interactions.

## 🎯 Features

- 👁️ Eye gaze-based mouse cursor movement.
- 😉 Wink detection for left/right click.
- 👁️👁️ Double-eye blink for double click.
- ✋ Hand gestures for switching modes and exiting:
  - **Open palm**: Start eye tracking.
  - **Index finger only**: Stop eye tracking.
  - **Index + middle fingers**: Exit application.

## 🧰 Tech Stack

- Python 3.9+
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/eye-cursor-control.git
   cd eye-cursor-control
