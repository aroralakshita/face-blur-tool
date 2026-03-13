# Face Blur Tool — Real-Time Identity Protection

A command-line tool for real-time face detection and anonymization from webcam or video input. Built with MediaPipe and OpenCV, designed for CPU-only deployment.

```bash
python main.py --input video.mp4 --blur 101 --confidence 0.6 --output anonymized.mp4
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Performance](#performance)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Extending the Tool](#extending-the-tool)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

**Requirements:** Python 3.8+, webcam or video file, Windows/Linux/macOS

```bash
# 1. Clone and enter project
git clone https://github.com/yourusername/face-blur-tool
cd face-blur-tool

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
cd src
python main.py
```

---

## Architecture

The pipeline processes each frame through four sequential layers:

```
Capture → Detect/Track → Blur → Display
```

**Detection Layer** (`detectors/`) — Runs face detection every N frames using MediaPipe. Full detection is expensive, so it doesn't run on every frame.

**Tracking Layer** (`trackers/`) — Between detections, faces are tracked using IoU (Intersection over Union) matching. IoU measures bounding box overlap between the current and previous frame. If overlap exceeds a threshold, it's the same face — no re-detection needed. This is what makes the tool run smoothly at real-time speeds.

**Processing Layer** (`filters/`) — Applies Gaussian blur to detected face regions. The blur region is expanded beyond the raw bounding box by a configurable pixel margin to account for detection inaccuracy at the edges.

**Presentation Layer** (`utils/overlay.py`) — Renders FPS, face count, and detection/tracking status onto the output frame before display.

### Detector Interface

All detectors implement an abstract base class (`detectors/base_detector.py`):

```python
class BaseDetector(ABC):
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        ...
    def close(self) -> None:
        ...
```

The rest of the pipeline only depends on this interface — not on MediaPipe. This means swapping in a different detection backend (YOLO, Haar cascades) requires creating one new file and zero changes elsewhere.

---

## Design Decisions

**Why MediaPipe over heavier models (MTCNN, RetinaFace)?**
MediaPipe runs efficiently on CPU without a GPU. Heavier CNN-based detectors are more accurate under challenging conditions (low light, extreme angles) but require significantly more compute. For real-time webcam anonymization on consumer hardware, MediaPipe's tradeoff is the right call.

**Why detection interval instead of per-frame detection?**
Running full detection on every frame at 1280×720 would cap performance around 10–15 FPS on most CPUs. Detecting every 5 frames.

**Why IoU tracking instead of a learned tracker (e.g. SORT, DeepSORT)?**
For a single-camera, low-motion use case (person sitting in front of webcam), IoU matching is sufficient and adds no additional dependencies. Learned trackers would improve accuracy under fast movement or occlusion at the cost of complexity.

**Why separate config.py and CLI args?**
`config.py` holds stable defaults committed to version control. CLI args are ephemeral runtime overrides. This pattern avoids polluting git history with tuning changes and makes the tool scriptable without editing source files.

**Why Gaussian blur over pixelation?**
Gaussian blur is simpler, faster, and harder to reverse than low-resolution pixelation. A pixelation mode (`--mode pixelate`) is a planned addition.

---

## Performance

Performance varies by hardware. MediaPipe face detection on CPU is the primary bottleneck — the detection interval and IoU tracking system exist specifically to reduce how often it runs.

Use benchmark mode to measure performance on your own machine:
```bash
python main.py --benchmark                  # 300 frames at default settings
python main.py --benchmark --frames 500     # longer run
python main.py --benchmark --interval 10    # test a specific config
```
Results include avg, min, max, and P5 FPS. P5 (5th percentile) is reported alongside minimum because the absolute minimum is often a single outlier frame — P5 better reflects sustained worst-case performance.

---

## CLI Reference

```
python main.py [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` | path | webcam | Video file path. Omit to use webcam. |
| `--output` | path | none | Save processed video to file (e.g. `out.mp4`) |
| `--confidence` | float | 0.5 | Detection confidence threshold (0.0–1.0) |
| `--blur` | int | 201 | Blur kernel size. Must be odd. Higher = stronger blur. |
| `--interval` | int | 5 | Frames between full detections. Lower = more accurate, higher CPU. |
| `--benchmark` | flag | off | Run benchmark mode and exit after N frames. |
| `--frames` | int | 300 | Frame count for benchmark mode. |

**Examples:**

```bash
# Process a recorded video and save output
python main.py --input meeting.mp4 --output anonymized.mp4

# Webcam with lighter blur and stricter detection
python main.py --blur 51 --confidence 0.7

# Benchmark at a non-default interval
python main.py --benchmark --interval 3 --frames 500
```

**Controls (live mode):**
- `ESC` or `q` — exit

---

## Configuration

Edit `src/config.py` to change defaults that apply when no CLI flag is passed.

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_INDEX` | 0 | Webcam device index |
| `CAMERA_WIDTH` | 1280 | Capture width |
| `CAMERA_HEIGHT` | 720 | Capture height |
| `DETECTION_CONFIDENCE` | 0.5 | Minimum detection confidence |
| `DETECTION_INTERVAL` | 5 | Frames between full detections |
| `BLUR_STRENGTH` | 201 | Blur kernel size (must be odd) |
| `BLUR_EXPANSION` | 15 | Pixels to expand face region beyond bounding box |
| `TRACKING_SMOOTHING` | 0.3 | Bounding box smoothing factor (0–1) |
| `MODEL_SELECTION` | 0 | 0 = short-range ≤2m (faster), 1 = full-range ≤5m |

---

## Project Structure

```
face-blur-tool/
├── requirements.txt
├── README.md
└── src/
    ├── main.py               # Entry point, CLI parsing, application loop
    ├── config.py             # Default configuration constants
    ├── detectors/
    │   ├── base_detector.py  # Abstract detector interface (BaseDetector)
    │   └── face_detector.py  # MediaPipe implementation of BaseDetector
    ├── trackers/
    │   └── face_tracker.py   # IoU-based face tracking between detections
    ├── filters/
    │   └── blur_filter.py    # Gaussian blur application with expansion
    └── utils/
        ├── fps_counter.py    # Rolling FPS calculation
        ├── overlay.py        # On-screen UI rendering
        └── benchmark.py      # FPS stats collection and reporting
```

---

## Extending the Tool

### Adding a new detector backend

1. Create `src/detectors/your_detector.py`
2. Inherit from `BaseDetector` and implement `detect()` and `close()`
3. Pass it into `FaceBlurApplication` — nothing else in the codebase changes

```python
from detectors.base_detector import BaseDetector

class YOLODetector(BaseDetector):
    def detect(self, frame):
        # your implementation
        return boxes  # list of (x, y, w, h)

    def close(self):
        pass
```

This is the Open/Closed Principle in practice: the system is open for extension (new detector) and closed for modification (no changes to main.py, tracker, or filters).

---

## Troubleshooting

**Camera not found**
Try `--input 1` or `--input 2` to use a different webcam index. On Linux, check `ls /dev/video*`.

**Low FPS**
Lower camera resolution in `config.py`, increase `--interval`, or close other applications using the camera.

**Faces not detected**
Ensure good lighting. Try lowering `--confidence` to 0.3. Switch to `MODEL_SELECTION = 1` in `config.py` for full-range detection.

**Blur kernel error**
`--blur` must be an odd number. The tool will auto-correct by incrementing by 1 and log a warning.

---

## Dependencies

- [MediaPipe](https://mediapipe.dev/) — Face detection
- [OpenCV](https://opencv.org/) — Frame capture and image processing

## License

Provided for educational and personal use.