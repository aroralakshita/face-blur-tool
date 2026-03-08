"""
MediaPipe Tasks API face detector implementation.

Implements the BaseDetector interface using MediaPipe's Tasks API

MediaPipe was chosen over heavier CNN models (MTCNN, RetinaFace)
because it runs efficiently on CPU, which is important for real-time webcam
use without requiring a GPU.

Requires the TFLite model file:
    src/blaze_face_short_range.tflite
Download from: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

Tradeoff: MediaPipe is fast and lightweight but less accurate than
heavier models in challenging lighting or extreme angles. Acceptable
for the real-time identity protection use case.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from detectors.base_detector import BaseDetector


# Model file path (relative to this file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'blaze_face_short_range.tflite')


@dataclass
class BoundingBox:
    """Represents a bounding box around a detected face.
    
    Attributes:
        x: Top-left x coordinate in pixels
        y: Top-left y coordinate in pixels
        width: Width of the bounding box in pixels
        height: Height of the bounding box in pixels
        confidence: Detection confidence score (0-1)
    """
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    def to_tuple(self) -> tuple:
        """Convert to tuple format (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def to_roi(self) -> tuple:
        """Convert to ROI format (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class FaceDetector(BaseDetector):
    """Face detector using MediaPipe Tasks API.

    Inherits from BaseDetector, fulfilling the interface contract that
    main.py and the rest of the pipeline depend on. The pipeline only
    calls .detect() and .close() — it has no knowledge of MediaPipe.

    Args:
        detection_confidence: Minimum confidence threshold (0.0-1.0).
            Lower values detect more faces but increase false positives.
        model_path: Path to the TFLite model file (uses default if None).
    """
    
    def __init__(
        self,
        detection_confidence: float = 0.5,
        model_path: Optional[str] = None
    ):
        
        self.detection_confidence = detection_confidence
        self.model_path = model_path or os.path.abspath(MODEL_PATH)
        
        # Initialize MediaPipe Face Detector using Tasks API
        self._FaceDetector = mp.tasks.vision.FaceDetector
        self._FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self._RunningMode = mp.tasks.vision.RunningMode
        self._BaseOptions = mp.tasks.BaseOptions
        
        # Create options
        options = self._FaceDetectorOptions(
            base_options=self._BaseOptions(model_asset_path=self.model_path),
            running_mode=self._RunningMode.IMAGE,
            min_detection_confidence=detection_confidence
        )
        
        # Create detector
        self._detector = self._FaceDetector.create_from_options(options)
    
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect faces in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of BoundingBox objects for each detected face
        """
        if frame is None or frame.size == 0:
            return []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform detection
        results = self._detector.detect(mp_image)
        
        boxes = []
        if results.detections:
            frame_height, frame_width = frame.shape[:2]
            
            for detection in results.detections:
                # Extract bounding box (now in absolute coordinates)
                bbox = detection.bounding_box
                
                # Get coordinates
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                width = int(bbox.width)
                height = int(bbox.height)
                
                # Get confidence score
                confidence = detection.categories[0].score if detection.categories else 0.0
                
                # Clamp to frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, frame_width - x)
                height = min(height, frame_height - y)
                
                # Create bounding box
                box = BoundingBox(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence
                )
                boxes.append(box)
        
        return boxes
    
    def close(self) -> None:
        """Release detector resources."""
        self._detector.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
