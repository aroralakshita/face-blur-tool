"""
Main application entry point for the Face Blur Tool.

This module orchestrates the face detection, tracking, and blurring
pipeline for real-time identity protection.
"""

import sys
import cv2
import logging
import numpy as np

from config import Config
from detectors.face_detector import FaceDetector
from trackers.face_tracker import FaceTracker
from filters.blur_filter import BlurFilter
from utils.fps_counter import FPSCounter
from utils.overlay import OverlayRenderer


# Using logging instead of print() throughout because:
# - Log levels (INFO, WARNING, ERROR) make severity explicit
# - Can redirect to file without changing code
# - Can suppress INFO in production by changing log level
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, config: Config = None):
        """Initialize the application.
        
        Args:
            config: Application configuration (uses default if None)
        """
        self.config = config or Config()
        
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
    
    def initialize(self) -> bool:
        """Initialize the webcam capture.
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Initialize webcam
        self._cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        
        if not self._cap.isOpened():
            logger.error("Could not open webcam")
            return False
        
        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        # Create display window
        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.config.WINDOW_NAME,
            self.config.WINDOW_WIDTH,
            self.config.WINDOW_HEIGHT
        )
        
        print(f"Initialized webcam: {self.config.CAMERA_WIDTH}x{self.config.CAMERA_HEIGHT}")
        print(f"Press ESC or 'q' to exit")
        
        return True
    
    def run(self) -> None:
        """Run the main application loop."""
        if not self.initialize():
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
                
                # Update FPS counter
                self.fps_counter.tick()
                
                # Determine if we should perform detection
                is_detecting = self.tracker.should_detect(self._frame_count)
                
                # Face detection/tracking
                if is_detecting:
                    boxes = self.detector.detect(frame)
                    self.tracker.update(boxes, self._frame_count)
                else:
                    boxes = self.tracker.get_tracked()
                
                # Apply blur to detected faces
                frame = self.blur_filter.apply_blur(frame, boxes)
                
                # Render overlays
                frame = self.overlay.render_all(
                    frame,
                    self.fps_counter.get_display_text(),
                    len(boxes),
                    is_detecting
                )
                
                # Display the frame
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
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        # Release detector resources
        self.detector.close()
        
        # Release webcam
        if self._cap is not None:
            self._cap.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    print("=" * 50)
    print("Face Blur Tool - Identity Protection")
    print("=" * 50)
    
    app = FaceBlurApplication()
    app.run()
    
    print("Application terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
