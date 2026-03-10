"""
Main application entry point for the Face Blur Tool.

This module orchestrates the face detection, tracking, and blurring
pipeline for real-time identity protection.

CLI Usage:
    python main.py                          # webcam with defaults
    python main.py --input video.mp4        # process a video file
    python main.py --output out.mp4         # save output to file
    python main.py --confidence 0.6         # stricter detection
    python main.py --blur 101               # lighter blur
    python main.py --interval 3             # detect every 3 frames
    python main.py --benchmark              # run benchmark mode (300 frames)
    python main.py --benchmark --frames 500 # benchmark with custom frame count
"""

import sys
import cv2
import argparse
import logging
import numpy as np

from config import Config
from detectors.face_detector import FaceDetector
from trackers.face_tracker import FaceTracker
from filters.blur_filter import BlurFilter
from utils.fps_counter import FPSCounter
from utils.overlay import OverlayRenderer
from utils.benchmark import BenchmarkRecorder


# Using logging instead of print() throughout because:
# - Log levels (INFO, WARNING, ERROR) make severity explicit
# - Can redirect to file without changing code
# - Can suppress INFO in production by changing log level
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    CLI args override config.py defaults at runtime. This lets config.py
    remain stable in version control while allowing flexible usage.

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Face Blur Tool — real-time identity protection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # shows defaults in --help
    )

    # Input/output
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input source. Omit for webcam, or pass path to video file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save processed video (e.g. output.mp4)"
    )

    # Detection settings
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Minimum detection confidence (0.0–1.0). Lower = more detections, more false positives."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Frames between full detections. Lower = more accurate, higher CPU usage."
    )

    # Blur settings
    parser.add_argument(
        "--blur",
        type=int,
        default=None,
        help="Blur kernel size (must be odd number). Higher = stronger blur."
    )

    # Benchmark mode
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode: measures FPS over N frames then exits."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames to run in benchmark mode."
    )

    return parser.parse_args()

def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI argument overrides to config object.

    Config.py holds stable defaults. CLI args are runtime overrides.
    We only override fields where the user explicitly passed a value.

    Args:
        config: Default configuration object
        args: Parsed CLI arguments

    Returns:
        Modified config with overrides applied
    """
    if args.confidence is not None:
        logger.info(f"Overriding confidence: {config.DETECTION_CONFIDENCE} → {args.confidence}")
        config.DETECTION_CONFIDENCE = args.confidence

    if args.interval is not None:
        logger.info(f"Overriding detection interval: {config.DETECTION_INTERVAL} → {args.interval}")
        config.DETECTION_INTERVAL = args.interval

    if args.blur is not None:
        # Enforce odd number requirement for Gaussian kernel
        blur = args.blur if args.blur % 2 == 1 else args.blur + 1
        if blur != args.blur:
            logger.warning(f"Blur kernel must be odd. Adjusted {args.blur} → {blur}")
        logger.info(f"Overriding blur strength: {config.BLUR_STRENGTH} → {blur}")
        config.BLUR_STRENGTH = blur

    return config


class FaceBlurApplication:
    """Main application class for the Face Blur Tool.
    
    This class manages the application lifecycle and coordinates
    all components for real-time face detection and blurring.
    
    Attributes:
        config: Application configuration
        detector: Face detection module
        tracker: Face tracking module
        blur_filter: Blur filter module
        fps_counter: FPS counter utility
        overlay: Overlay renderer
    """
    
    def __init__(self, config: Config = None, args: argparse.Namespace = None):

        self.config = config or Config()
        self.args = args
        
        # Initialize components
        self.detector = FaceDetector(
            detection_confidence=self.config.DETECTION_CONFIDENCE
        )
        self.tracker = FaceTracker(
            smoothing_factor=self.config.TRACKING_SMOOTHING,
            detection_interval=self.config.DETECTION_INTERVAL
        )
        self.blur_filter = BlurFilter(
            kernel_size=self.config.BLUR_STRENGTH,
            expansion=self.config.BLUR_EXPANSION
        )
        self.fps_counter = FPSCounter()
        self.overlay = OverlayRenderer(
            fps_position=self.config.FPS_POSITION,
            fps_font_scale=self.config.FPS_FONT_SCALE,
            fps_color=self.config.FPS_COLOR,
            face_count_position=self.config.FACE_COUNT_POSITION,
            face_count_font_scale=self.config.FACE_COUNT_FONT_SCALE,
            face_count_color=self.config.FACE_COUNT_COLOR,
            status_position=self.config.STATUS_POSITION,
            status_font_scale=self.config.STATUS_FONT_SCALE,
            status_detecting_color=self.config.STATUS_DETECTING_COLOR,
            status_tracking_color=self.config.STATUS_TRACKING_COLOR
        )
        
        # Application state
        self._running = False
        self._frame_count = 0
        self._cap = None
        self.writer = None
    
    def initialize(self) -> bool:
        """Initialize video capture and optional output writer.
        
        Supports both webcam (default) and video file input via --input.

        Returns:
            True if initialization successful, False otherwise
        """
        #Determine input source
        #Integer index = webcam, string path = video file
        if self.args and self.args.input:
            source = self.args.input
            logger.info(f"Input: video file '{source}'")
        else:
            source = self.config.CAMERA_INDEX
            logger.info(f"Input: webcam index {source}")
        
        self._cap = cv2.VideoCapture(source)
        
        if not self._cap.isOpened():
            logger.error(f"Could not open inout source: {source}")
            return False
        
        # Only set camera properties for webcam (int source)
        # For video files these would be ignored anyway
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        # Set up output writer if --output was specified
        if self.args and self.args.output:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.args.output, fourcc, fps, (width, height))
            logger.info(f"Output: saving to '{self.args.output}' at {width}x{height} {fps:.0f}fps")
        
        # Create display window (skip in benchmark mode for clean output)
        if not (self.args and self.args.benchmark):
            cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.config.WINDOW_NAME,
                self.config.WINDOW_WIDTH,
                self.config.WINDOW_HEIGHT
        )
        
        logger.info(f"Press ESC or 'q' to exit")
        
        return True
    
    def run(self) -> None:
        """Run the main application loop."""
        if not self.initialize():
            return
        
        if self.args and self.args.benchmark:
            self._run_benchmark(self.args.frames)
            return
        
        self._running = True
        self._frame_count = 0
        
        try:
            while self._running:
                # Capture frame
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("End of stream or failed to read frame from webcam")
                    break

                frame = self._process_frame(frame)
                            
                cv2.imshow(self.config.WINDOW_NAME, frame)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if key in self.config.EXIT_KEYS:
                    self._running = False
                
                self._frame_count += 1
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.cleanup()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the full pipeline.

        Separated from the main loop so benchmark mode can reuse it.
        This is the core pipeline:
            capture → detect/track → blur → overlay

        Args:
            frame: Raw BGR frame from capture device

        Returns:
            Processed frame with faces blurred and overlays rendered
        """
        self.fps_counter.tick()

        is_detecting = self.tracker.should_detect(self._frame_count)

        if is_detecting:
            boxes = self.detector.detect(frame)
            self.tracker.update(boxes, self._frame_count)
        else:
            boxes = self.tracker.get_tracked()

        frame = self.blur_filter.apply_blur(frame, boxes)
        frame = self.overlay.render_all(
            frame,
            self.fps_counter.get_display_text(),
            len(boxes),
            is_detecting
        )

        return frame
    
    def _run_benchmark(self, num_frames: int) -> None:
        """Run benchmark mode: process N frames and report performance stats.

        Benchmark mode skips the display window and key handling to measure
        raw pipeline performance. Results are printed to stdout in a format
        suitable for documentation.

        Args:
            num_frames: Number of frames to process
        """
        logger.info(f"Starting benchmark: {num_frames} frames")

        # Get actual capture resolution (may differ from config if using video file)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width}x{height}"

        recorder = BenchmarkRecorder()

        try:
            for i in range(num_frames):
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning(f"Stream ended early at frame {i}")
                    break

                self.fps_counter.tick()
                self._process_frame(frame)

                current_fps = self.fps_counter.get_fps()
                if current_fps > 0:
                    recorder.record(current_fps)

                # Progress indicator every 50 frames
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i + 1}/{num_frames} frames processed...")

        finally:
            self.cleanup()

        # Print results
        recorder.print_results(resolution)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        # Release detector resources
        self.detector.close()
        
        # Release webcam
        if self._cap is not None:
            self._cap.release()
        
        if self._writer is not None:
            self._writer.release()
            logger.info("Output video saved")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    print("=" * 50)
    print("Face Blur Tool - Identity Protection")
    print("=" * 50)
    
    args = parse_args()
    config = Config()
    config = apply_cli_overrides(config, args)

    app = FaceBlurApplication(config=config, args=args)
    app.run()
    
    print("Application terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
