# Face Blur Tool - Identity Protection

A Python-based real-time face detection and anonymization tool that automatically blurs faces from webcam input to protect identity privacy.

## Features

- **Real-time webcam input processing**
- **MediaPipe-based face detection** - Accurate and fast face detection
- **Pixel blur filter** - Automatic face anonymization
- **Exit key handling** - Press ESC or 'q' to exit
- **FPS display** - Monitor performance in real-time
- **Face tracking** - Smooth tracking to avoid re-detection every frame

## Requirements

- Python 3.8 or higher
- Webcam (USB or built-in)
- Windows/Linux/macOS

## Installation

1. Clone or download this project:

```bash
cd face-blur-tool
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
cd src
python main.py
```

### Controls

- **ESC** or **q** - Exit the application

### On-Screen Information

- **FPS** - Current frames per second (top-left, green)
- **Faces** - Number of detected faces (top-left, cyan)
- **Status** - DETECTING or TRACKING mode (top-left, yellow/orange)

## Configuration

Edit `src/config.py` to customize settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `CAMERA_INDEX` | Webcam device index | 0 |
| `CAMERA_WIDTH` | Capture width | 1280 |
| `CAMERA_HEIGHT` | Capture height | 720 |
| `DETECTION_CONFIDENCE` | Minimum detection confidence | 0.5 |
| `DETECTION_INTERVAL` | Frames between detections | 5 |
| `BLUR_STRENGTH` | Blur kernel size (odd number) | 51 |
| `BLUR_EXPANSION` | Pixels to expand face region | 15 |
| `TRACKING_SMOOTHING` | Smoothing factor (0-1) | 0.3 |

## Project Structure

```
face-blur-tool/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── main.py               # Application entry point
    ├── config.py             # Configuration constants
    ├── detectors/
    │   ├── __init__.py
    │   └── face_detector.py  # MediaPipe face detection
    ├── trackers/
    │   ├── __init__.py
    │   └── face_tracker.py   # Face tracking across frames
    ├── filters/
    │   ├── __init__.py
    │   └── blur_filter.py    # Blur filter implementation
    └── utils/
        ├── __init__.py
        ├── fps_counter.py    # FPS calculation
        └── overlay.py        # UI overlay rendering
```

## How It Works

1. **Frame Capture**: Webcam frames are captured at the configured resolution
2. **Face Detection**: MediaPipe detects faces in the frame (every N frames)
3. **Face Tracking**: Between detections, faces are tracked using IoU matching
4. **Blur Application**: Gaussian blur is applied to detected face regions
5. **Display**: Processed frame with overlays is shown in the window

## Performance Tips

- Lower `DETECTION_INTERVAL` for more accurate tracking (higher CPU usage)
- Increase `BLUR_STRENGTH` for stronger blur (slightly slower)
- Use `MODEL_SELECTION = 0` for close-up webcam use (faster)
- Use `MODEL_SELECTION = 1` for room-scale detection (slower but more accurate)

## Troubleshooting

### Camera not found
- Ensure your webcam is connected
- Try changing `CAMERA_INDEX` to 1 or 2 in config.py

### Low FPS
- Lower the camera resolution in config.py
- Increase `DETECTION_INTERVAL`
- Close other applications using the camera

### Faces not detected
- Ensure good lighting
- Try lowering `DETECTION_CONFIDENCE`
- Use `MODEL_SELECTION = 1` for full-range detection

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Face detection
- [OpenCV](https://opencv.org/) - Image processing
