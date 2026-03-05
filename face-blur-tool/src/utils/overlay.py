"""
Overlay rendering utility for UI elements.

This module provides functionality to render UI overlays
such as FPS counter, face count, and status indicators.
"""

from typing import Tuple

import cv2
import numpy as np


class OverlayRenderer:
    """Renders UI overlay elements on frames.
    
    This class handles the rendering of various UI elements
    including FPS display, face count, and detection status.
    """
    
    def __init__(
        self,
        fps_position: Tuple[int, int] = (10, 30),
        fps_font_scale: float = 0.7,
        fps_color: Tuple[int, int, int] = (0, 255, 0),
        face_count_position: Tuple[int, int] = (10, 60),
        face_count_font_scale: float = 0.7,
        face_count_color: Tuple[int, int, int] = (255, 255, 0),
        status_position: Tuple[int, int] = (10, 90),
        status_font_scale: float = 0.6,
        status_detecting_color: Tuple[int, int, int] = (0, 255, 255),
        status_tracking_color: Tuple[int, int, int] = (255, 165, 0)
    ):
        """Initialize the overlay renderer.
        
        Args:
            fps_position: Position for FPS text (x, y)
            fps_font_scale: Font scale for FPS text
            fps_color: Color for FPS text (B, G, R)
            face_count_position: Position for face count text
            face_count_font_scale: Font scale for face count
            face_count_color: Color for face count text
            status_position: Position for status text
            status_font_scale: Font scale for status text
            status_detecting_color: Color when detecting
            status_tracking_color: Color when tracking
        """
        self.fps_position = fps_position
        self.fps_font_scale = fps_font_scale
        self.fps_color = fps_color
        self.face_count_position = face_count_position
        self.face_count_font_scale = face_count_font_scale
        self.face_count_color = face_count_color
        self.status_position = status_position
        self.status_font_scale = status_font_scale
        self.status_detecting_color = status_detecting_color
        self.status_tracking_color = status_tracking_color
        
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_thickness = 2
    
    def render_fps(
        self,
        frame: np.ndarray,
        fps_text: str
    ) -> np.ndarray:
        """Render FPS text on frame.
        
        Args:
            frame: Input BGR image
            fps_text: FPS text to display
            
        Returns:
            Frame with FPS overlay
        """
        cv2.putText(
            frame,
            fps_text,
            self.fps_position,
            self._font,
            self.fps_font_scale,
            self.fps_color,
            self._font_thickness,
            cv2.LINE_AA
        )
        return frame
    
    def render_face_count(
        self,
        frame: np.ndarray,
        count: int
    ) -> np.ndarray:
        """Render face count on frame.
        
        Args:
            frame: Input BGR image
            count: Number of faces detected
            
        Returns:
            Frame with face count overlay
        """
        text = f"Faces: {count}"
        cv2.putText(
            frame,
            text,
            self.face_count_position,
            self._font,
            self.face_count_font_scale,
            self.face_count_color,
            self._font_thickness,
            cv2.LINE_AA
        )
        return frame
    
    def render_status(
        self,
        frame: np.ndarray,
        is_detecting: bool
    ) -> np.ndarray:
        """Render detection status on frame.
        
        Args:
            frame: Input BGR image
            is_detecting: True if currently detecting, False if tracking
            
        Returns:
            Frame with status overlay
        """
        if is_detecting:
            text = "Status: DETECTING"
            color = self.status_detecting_color
        else:
            text = "Status: TRACKING"
            color = self.status_tracking_color
        
        cv2.putText(
            frame,
            text,
            self.status_position,
            self._font,
            self.status_font_scale,
            color,
            self._font_thickness,
            cv2.LINE_AA
        )
        return frame
    
    def render_all(
        self,
        frame: np.ndarray,
        fps_text: str,
        face_count: int,
        is_detecting: bool
    ) -> np.ndarray:
        """Render all overlay elements on frame.
        
        Args:
            frame: Input BGR image
            fps_text: FPS text to display
            face_count: Number of faces detected
            is_detecting: True if currently detecting
            
        Returns:
            Frame with all overlays
        """
        frame = self.render_fps(frame, fps_text)
        frame = self.render_face_count(frame, face_count)
        frame = self.render_status(frame, is_detecting)
        return frame
    
    def render_help_text(
        self,
        frame: np.ndarray,
        text: str = "Press ESC or 'q' to exit"
    ) -> np.ndarray:
        """Render help text at the bottom of the frame.
        
        Args:
            frame: Input BGR image
            text: Help text to display
            
        Returns:
            Frame with help text overlay
        """
        frame_height = frame.shape[0]
        position = (10, frame_height - 10)
        
        cv2.putText(
            frame,
            text,
            position,
            self._font,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        return frame
