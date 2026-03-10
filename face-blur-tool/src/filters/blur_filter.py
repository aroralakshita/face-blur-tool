"""
Blur filter module for face anonymization.

Applies Gaussian blur to detected face regions for identity protection.
Accepts boxes as plain (x, y, w, h) tuples
"""

import cv2
import numpy as np



class BlurFilter:
    """Applies Gaussian blur to face regions.
    
    This class handles the blurring of detected face regions in frames,
    including bounding box expansion and ROI validation.
    
    Attributes:
        kernel_size: Size of the Gaussian blur kernel (must be odd, larger = more blur).
            Auto-corrected to the next odd number if an even value is passed.
        expansion: Pixels to expand bounding box around face
    """
    
    def __init__(
        self,
        kernel_size: int = 51,
        expansion: int = 15
    ):

        # Ensure kernel size is odd and positive
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = max(1, kernel_size)
        self.expansion = expansion
    
    def apply_blur(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Apply blur to all detected face regions.
        
        Args:
            frame: Input BGR image as numpy array
            boxes: List of (x, y, w, h) bounding boxes
            
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
        box: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Blur a single face region.

        Expands the box, clamps to frame bounds, extracts the ROI,
        applies Gaussian blur, and writes it back in place
        
        Args:
            frame: Input BGR image
            box: (x, y, w, h) ounding box
            
        Returns:
            Frame with the face region blurred
        """
        frame_h, frame_w = frame.shape[:2]
        
        x1, y1, x2, y2 = self._expand_box(box, frame_w, frame_h)
        x1, y1, x2, y2 = self._clamp_roi(x1, y1, x2, y2, frame_w, frame_h)
        
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
        box: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int
    ) -> tuple[int, int, int, int]:
        """Expand bounding box by the expansion margin.
        
        Converts from (x, y, w, h) to (x1, y1, x2, y2) and expands
        outwards on all sides. Clamped to frame bounds
        
        Args:
            box: (x, y, w, h) bounding box
            frame_w: Width of the frame
            frame_h: Height of the frame
            
        Returns:
            Expanded (x1, y1, x2, y2) coordinates
        """
        x, y, w, h = box

        x1 = max(0, x - self.expansion)
        y1 = max(0, y - self.expansion)
        x2 = min(frame_w, x + w + self.expansion)
        y2 = min(frame_h, y + h + self.expansion)
        
        return x1, y1, x2, y2
    
    def _clamp_roi(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int
    ) -> tuple[int, int, int, int]:
        """Clamp ROI coordinates to frame bounds.
        
        Args:
            x1, y1, x2, y2: ROI coordinates
            frame_w: Width of the frame
            frame_h: Height of the frame
            
        Returns:
            Clamped ROI coordinates
        """
        x1 = max(0, min(x1, frame_w))
        y1 = max(0, min(y1, frame_h))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))
        
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
            expansion: New expansion margin in pixels. Clamped to minimum 0
        """
        self.expansion = max(0, expansion)
