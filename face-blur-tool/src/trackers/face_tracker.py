"""
Face tracking module for smooth face tracking across frames.

Tracks detected faces between full detection runs using IoU (Intersection
over Union) matching. This is what allows the tool to run at real-time
speeds. Full MediaPipe detection only runs every N frames, and this
module fills the gaps by matching existing face positions to new frames.

IoU measures bounding box overlap (0.0 = no overlap, 1.0 = identical).
If two boxes overlap above the threshold (0.3), they're considered the
same face. No re-detection needed.

All boxes are plain (x, y, w, h) tuples
"""

from dataclasses import dataclass


Box = tuple[int, int, int, int]


@dataclass
class TrackedFace:
    """Represents a tracked face between detection runs.
    
    Attributes:
        box: Current smoothed bounding box as (x, y, w, h)
        frames_since_detection: Frames since last detection update
    """
    box: Box
    frames_since_detection: int = 0


class FaceTracker:
    """Tracks faces across frames with IoU matching temporal smoothing.
    
    Between full detection runs, this class maintains face positions across frames and applies
    smoothing to reduce jitter. It also determines when re-detection
    is needed.
    
    Attributes:
        smoothing_factor: Controls how much the box moves per frame (0.0-1.0)
                          Higher values = more smoothing (slower response)
                          Default 0.3 gives a responsive but stable result
        detection_interval: Number of frames between full detections. Tracked faces
        that go unmatched for 2x this value are dropped
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.3,
        detection_interval: int = 5
    ):

        self.smoothing_factor = smoothing_factor
        self.detection_interval = detection_interval
        
        self._tracked_faces: list[TrackedFace] = []
        self._last_detection_frame = 0
    
    def update(
        self,
        new_boxes: list[Box],
        frame_count: int
    ) -> None:
        """Update tracked faces with new detections.
        
        This method matches new detections to existing tracked faces via IoU
        and applies smoothing to matched faces, expires unmatched ones,
        and adds any newly detected faces
        
        Args:
            new_boxes: List of (x, y, w, h) detected tuples
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
            self._tracked_faces = [TrackedFace(box=box) for box in new_boxes]
            return
        
        # Match new boxes to existing tracked faces
        matched_tracked = set()
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
               tracked.box = self._smooth_box(tracked.box, new_boxes[best_match_idx])
               tracked.frames_since_detection = 0
               
               matched_tracked.add(i)
               new_matched.add(best_match_idx)
        
        # Remove unmatched tracked faces
        self._tracked_faces = [
            t for i, t in enumerate(self._tracked_faces)
            if i in matched_tracked or t.frames_since_detection < self.detection_interval
        ]
        
        # Add new unmatched detections
        for j, new_box in enumerate(new_boxes):
            if j not in new_matched:
                self._tracked_faces.append(TrackedFace(box=new_box))
    
    def get_tracked(self) -> list[Box]:
        """Get currently tracked face positions.
        
        Returns:
            List of (x, y, w, h) tuples
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
        old_box: Box,
        new_box: Box
    ) -> Box:
        """Apply exponential moving average smoothing to bounding box.
        
        Blends the previous position with the new detection using teh
        smoothing factor as the weight on teh old position

        Formula: smoothed = (old * alpha) + (new * (1 - aplha))
        
        Args:
            old_box: Previous smoothed position (x, y, w, h)
            new_box: New detected position (x, y, w, h)
            
        Returns:
            Smoothed (x, y, w, h) tuple
        """
        # Exponential moving average smoothing
        alpha = self.smoothing_factor
        ox, oy, ow, oh = old_box
        nx, ny, nw, nh = new_box
        
        return (
            int(ox * alpha + nx * (1 - alpha)),
            int(oy * alpha + ny * (1 - alpha)),
            int(ow * alpha + nw * (1 - alpha)),
            int(oh * alpha + nh * (1 - alpha)),
        )
    
    def _calculate_iou(
        self,
        box1: Box,
        box2: Box
    ) -> float:
        """Calculate Intersection over Union between two bounding boxes.
        
        IoU = intersection area / union area
        Returns 0.0 if boxes don't overlap
        Args:
            box1: First box (x, y, w, h)
            box2: Second box as (x, y, w, h)
            
        Returns:
            IoU score (0.0-1.0)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection rectangle
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        # Check if boxes overlap
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        # Calculate areas
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self) -> None:
        """Reset all tracked faces."""
        self._tracked_faces = []
        self._last_detection_frame = 0
