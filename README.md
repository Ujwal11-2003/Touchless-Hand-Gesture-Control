# Hand Gesture Recognition System

This is a full-stack web application that uses computer vision to detect and recognize hand gestures in real-time. The system can identify various hand gestures and display them on a modern web interface.

## Features

- Real-time hand gesture detection using your webcam
- Support for multiple gestures:
  - Open Palm
  - Peace Sign
  - Thumbs Up
  - Fist
- Modern, responsive web interface
- Real-time gesture updates using WebSocket

## Requirements

- Python 3.7 or higher
- Webcam
- Modern web browser

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Allow camera access when prompted by your browser.

4. Make hand gestures in front of your camera to see them detected in real-time.

## Supported Gestures

- **Open Palm**: All fingers extended
- **Peace**: Index and middle fingers extended
- **Thumbs Up**: Only thumb extended
- **Fist**: All fingers closed

## Technical Details

- Backend: Python with Flask and Flask-SocketIO
- Computer Vision: OpenCV and MediaPipe
- Frontend: HTML, JavaScript, and Tailwind CSS
- Real-time Communication: WebSocket

## License

This project is licensed under the MIT License - see the LICENSE file for details. 