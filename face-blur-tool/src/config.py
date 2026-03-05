"""
Configuration constants for the Face Blur Tool.
"""


class Config:
    """Application configuration settings."""
    
    # Window settings
    WINDOW_NAME = "Face Blur Tool - Press ESC or 'q' to exit"
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    
    # Camera settings
    CAMERA_INDEX = 0  # Default webcam
    CAMERA_FPS = 30
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    
    # Detection settings
    DETECTION_CONFIDENCE = 0.5
    TRACKING_CONFIDENCE = 0.5
    MODEL_SELECTION = 0  # 0 for short-range (faster), 1 for full-range
    
    # Blur settings
    BLUR_STRENGTH = 201    # Kernel size (must be odd number)
    BLUR_EXPANSION = 15   # Pixels to expand bounding box around face
    
    # Performance settings
    DETECTION_INTERVAL = 5  # Frames between full detections
    TRACKING_SMOOTHING = 0.3  # Smoothing factor for bounding box (0-1)
    
    # UI settings
    FPS_POSITION = (10, 30)
    FPS_FONT_SCALE = 0.7
    FPS_FONT_THICKNESS = 2
    FPS_COLOR = (0, 255, 0)  # Green
    
    FACE_COUNT_POSITION = (10, 60)
    FACE_COUNT_FONT_SCALE = 0.7
    FACE_COUNT_FONT_THICKNESS = 2
    FACE_COUNT_COLOR = (255, 255, 0)  # Cyan
    
    STATUS_POSITION = (10, 90)
    STATUS_FONT_SCALE = 0.6
    STATUS_FONT_THICKNESS = 2
    STATUS_DETECTING_COLOR = (0, 255, 255)  # Yellow
    STATUS_TRACKING_COLOR = (255, 165, 0)  # Orange
    
    # Exit keys
    EXIT_KEYS = [27, ord('q'), ord('Q')]  # ESC, q, Q
