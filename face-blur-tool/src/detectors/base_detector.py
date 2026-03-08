"""
Abstract base class for face detectors.

This module defines the contract that all detector implementations
must follow. This allows main.py to remain detector-agnostic —
it only depends on this interface, not on MediaPipe or any other
specific library.

Design pattern: Strategy Pattern
Any class that inherits from BaseDetector and implements detect()
and close() can be used as a drop-in replacement.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """Abstract interface for face detection implementations.

    All detectors used in this system must implement this interface.
    This decouples the detection algorithm from the rest of the pipeline.

    To add a new detector (e.g. YOLO):
        1. Create yolo_detector.py in this folder
        2. Inherit from BaseDetector
        3. Implement detect() and close()
        4. Pass it into FaceBlurApplication — nothing else changes

    Example:
        detector = YOLODetector(confidence=0.5)
        boxes = detector.detect(frame)  # same call, different backend
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in a single frame.

        Args:
            frame: BGR image as numpy array (standard OpenCV format)

        Returns:
            List of bounding boxes as (x, y, w, h) tuples in pixel coordinates.
            Returns empty list if no faces detected.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the detector.

        Called during application cleanup. Implement even if your
        detector has nothing to release — keeps the interface consistent.
        """
        pass