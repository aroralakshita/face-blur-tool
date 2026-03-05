"""
FPS counter utility for performance monitoring.

This module provides functionality to calculate and display
frames per second for real-time applications.
"""

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Calculates frames per second using a moving average.
    
    This class tracks frame timestamps and calculates FPS
    over a configurable window for stable display.
    
    Attributes:
        window_size: Number of frames to average for FPS calculation
    """
    
    def __init__(self, window_size: int = 30):
        """Initialize the FPS counter.
        
        Args:
            window_size: Number of frames to include in moving average
        """
        self.window_size = window_size
        self._frame_times: Deque[float] = deque(maxlen=window_size)
        self._last_time: float = time.perf_counter()
        self._fps: float = 0.0
        self._frame_count: int = 0
    
    def tick(self) -> None:
        """Record a frame timestamp.
        
        This method should be called once per frame.
        """
        current_time = time.perf_counter()
        self._frame_times.append(current_time)
        self._last_time = current_time
        self._frame_count += 1
    
    def get_fps(self) -> float:
        """Calculate current FPS.
        
        Returns:
            Current frames per second
        """
        if len(self._frame_times) < 2:
            return 0.0
        
        # Calculate FPS from the time span of the window
        time_span = self._frame_times[-1] - self._frame_times[0]
        if time_span > 0:
            self._fps = (len(self._frame_times) - 1) / time_span
        
        return self._fps
    
    def get_display_text(self) -> str:
        """Get formatted FPS text for display.
        
        Returns:
            Formatted FPS string (e.g., "FPS: 30.5")
        """
        fps = self.get_fps()
        return f"FPS: {fps:.1f}"
    
    def get_frame_count(self) -> int:
        """Get total frame count.
        
        Returns:
            Total number of frames processed
        """
        return self._frame_count
    
    def reset(self) -> None:
        """Reset the FPS counter."""
        self._frame_times.clear()
        self._last_time = time.perf_counter()
        self._fps = 0.0
        self._frame_count = 0
