"""
Face tracking module for smooth face tracking across frames.

This module provides functionality to track detected faces across frames,
reducing the need for full detection on every frame and providing
smoother bounding box transitions.
"""

from typing import List, Optional
from dataclasses import dataclass

from detectors.face_detector import BoundingBox


@dataclass
class TrackedFace:
    """Represents a tracked face with smoothing.
    
    Attributes:
        box: Current smoothed bounding box
        original_box: Last detected bounding box
        frames_since_detection: Frames since last detection update
    """
    box: BoundingBox
    original_box: BoundingBox
    frames_since_detection: int = 0


class FaceTracker:
    """Tracks faces across frames with temporal smoothing.
    
    This class maintains face positions across frames and applies
    smoothing to reduce jitter. It also determines when re-detection
    is needed.
    
    Attributes:
        smoothing_factor: Factor for temporal smoothing (0-1)
                          Higher values = more smoothing (slower response)
        detection_interval: Number of frames between full detections
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.3,
        detection_interval: int = 5
    ):
        """Initialize the face tracker.
        
        Args:
            smoothing_factor: Smoothing factor for bounding box (0-1)
                             0 = no smoothing, 1 = maximum smoothing
            detection_interval: Frames between full detections
        """
        self.smoothing_factor = smoothing_factor
        self.detection_interval = detection_interval
        
        self._tracked_faces: List[TrackedFace] = []
        self._last_detection_frame = 0
    
    def update(
        self,
        new_boxes: List[BoundingBox],
        frame_count: int
    ) -> None:
        """Update tracked faces with new detections.
        
        This method matches new detections to existing tracked faces
        and applies smoothing to bounding box positions.
        
        Args:
            new_boxes: Newly detected bounding boxes
            frame_count: Current frame number
        """
        self._last_detection_frame = frame_count
        
        if not new_boxes:
            # No detections - increment frame counter for tracked faces
            for tracked in self._tracked_faces:
                tracked.frames_since_detection += 1
            
            # Remove faces that haven't been detected for too long
            self._tracked_faces = [
                t for t in self._tracked_faces
                if t.frames_since_detection < self.detection_interval * 2
            ]
            return
        
        if not self._tracked_faces:
            # No existing tracked faces - add all new detections
            for box in new_boxes:
                self._tracked_faces.append(TrackedFace(
                    box=box,
                    original_box=box,
                    frames_since_detection=0
                ))
            return
        
        # Match new boxes to existing tracked faces
        matched_indices = set()
        new_matched = set()
        
        # Simple matching based on IoU (Intersection over Union)
        for i, tracked in enumerate(self._tracked_faces):
            best_match_idx = -1
            best_iou = 0.3  # Minimum IoU threshold
            
            for j, new_box in enumerate(new_boxes):
                if j in new_matched:
                    continue
                
                iou = self._calculate_iou(tracked.box, new_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j
            
            if best_match_idx >= 0:
                # Update tracked face with smoothed position
                new_box = new_boxes[best_match_idx]
                smoothed_box = self._smooth_box(tracked.box, new_box)
                
                tracked.box = smoothed_box
                tracked.original_box = new_box
                tracked.frames_since_detection = 0
                
                matched_indices.add(i)
                new_matched.add(best_match_idx)
        
        # Remove unmatched tracked faces
        self._tracked_faces = [
            t for i, t in enumerate(self._tracked_faces)
            if i in matched_indices or t.frames_since_detection < self.detection_interval
        ]
        
        # Add new unmatched detections
        for j, new_box in enumerate(new_boxes):
            if j not in new_matched:
                self._tracked_faces.append(TrackedFace(
                    box=new_box,
                    original_box=new_box,
                    frames_since_detection=0
                ))
    
    def get_tracked(self) -> List[BoundingBox]:
        """Get currently tracked face bounding boxes.
        
        Returns:
            List of smoothed bounding boxes for tracked faces
        """
        return [tracked.box for tracked in self._tracked_faces]
    
    def should_detect(self, frame_count: int) -> bool:
        """Determine if re-detection should be performed.
        
        Args:
            frame_count: Current frame number
            
        Returns:
            True if detection should be performed, False otherwise
        """
        # Always detect if no faces are tracked
        if not self._tracked_faces:
            return True
        
        # Detect at regular intervals
        frames_since_detection = frame_count - self._last_detection_frame
        return frames_since_detection >= self.detection_interval
    
    def _smooth_box(
        self,
        old_box: BoundingBox,
        new_box: BoundingBox
    ) -> BoundingBox:
        """Apply temporal smoothing to bounding box.
        
        Args:
            old_box: Previous smoothed bounding box
            new_box: New detected bounding box
            
        Returns:
            Smoothed bounding box
        """
        # Exponential moving average smoothing
        alpha = self.smoothing_factor
        
        return BoundingBox(
            x=int(old_box.x * alpha + new_box.x * (1 - alpha)),
            y=int(old_box.y * alpha + new_box.y * (1 - alpha)),
            width=int(old_box.width * alpha + new_box.width * (1 - alpha)),
            height=int(old_box.height * alpha + new_box.height * (1 - alpha)),
            confidence=new_box.confidence
        )
    
    def _calculate_iou(
        self,
        box1: BoundingBox,
        box2: BoundingBox
    ) -> float:
        """Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection coordinates
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        # Check if boxes overlap
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self) -> None:
        """Reset all tracked faces."""
        self._tracked_faces = []
        self._last_detection_frame = 0
