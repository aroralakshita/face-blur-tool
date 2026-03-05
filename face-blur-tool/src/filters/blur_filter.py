"""
Blur filter module for face anonymization.

This module provides functionality to apply pixel blur effects
to detected face regions for identity protection.
"""

from typing import List, Tuple

import cv2
import numpy as np

from detectors.face_detector import BoundingBox


class BlurFilter:
    """Applies blur effect to face regions.
    
    This class handles the blurring of detected face regions in frames,
    including bounding box expansion and ROI validation.
    
    Attributes:
        kernel_size: Size of the Gaussian blur kernel (must be odd)
        expansion: Pixels to expand bounding box around face
    """
    
    def __init__(
        self,
        kernel_size: int = 51,
        expansion: int = 15
    ):
        """Initialize the blur filter.
        
        Args:
            kernel_size: Size of Gaussian blur kernel (must be odd, larger = more blur)
            expansion: Pixels to expand bounding box around face
        """
        # Ensure kernel size is odd and positive
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = max(1, kernel_size)
        self.expansion = expansion
    
    def apply_blur(
        self,
        frame: np.ndarray,
        boxes: List[BoundingBox]
    ) -> np.ndarray:
        """Apply blur to all detected face regions.
        
        Args:
            frame: Input BGR image
            boxes: List of bounding boxes for faces
            
        Returns:
            Frame with blurred face regions
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Create a copy to avoid modifying the original
        result = frame.copy()
        
        for box in boxes:
            result = self._blur_region(result, box)
        
        return result
    
    def _blur_region(
        self,
        frame: np.ndarray,
        box: BoundingBox
    ) -> np.ndarray:
        """Blur a single face region.
        
        Args:
            frame: Input BGR image
            box: Bounding box for the face
            
        Returns:
            Frame with the face region blurred
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Expand the bounding box
        expanded_box = self._expand_box(box, frame_width, frame_height)
        
        # Extract ROI coordinates
        x1, y1, x2, y2 = expanded_box
        
        # Validate ROI is within frame bounds
        x1, y1, x2, y2 = self._validate_roi(x1, y1, x2, y2, frame_width, frame_height)
        
        # Check if ROI is valid
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Extract the region of interest
        roi = frame[y1:y2, x1:x2]
        
        # Apply Gaussian blur
        blurred_roi = cv2.GaussianBlur(roi, (self.kernel_size, self.kernel_size), 0)
        
        # Replace the original region with the blurred version
        frame[y1:y2, x1:x2] = blurred_roi
        
        return frame
    
    def _expand_box(
        self,
        box: BoundingBox,
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by the expansion margin.
        
        Args:
            box: Original bounding box
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Tuple of (x1, y1, x2, y2) for expanded box
        """
        x1 = max(0, box.x - self.expansion)
        y1 = max(0, box.y - self.expansion)
        x2 = min(frame_width, box.x + box.width + self.expansion)
        y2 = min(frame_height, box.y + box.height + self.expansion)
        
        return (x1, y1, x2, y2)
    
    def _validate_roi(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int, int, int]:
        """Validate and clamp ROI coordinates to frame bounds.
        
        Args:
            x1, y1, x2, y2: ROI coordinates
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Clamped ROI coordinates
        """
        x1 = max(0, min(x1, frame_width))
        y1 = max(0, min(y1, frame_height))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))
        
        return (x1, y1, x2, y2)
    
    def set_kernel_size(self, kernel_size: int) -> None:
        """Set the blur kernel size.
        
        Args:
            kernel_size: New kernel size (will be made odd if even)
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = max(1, kernel_size)
    
    def set_expansion(self, expansion: int) -> None:
        """Set the bounding box expansion margin.
        
        Args:
            expansion: New expansion margin in pixels
        """
        self.expansion = max(0, expansion)
