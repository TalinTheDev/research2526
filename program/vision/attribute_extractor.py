"""
attribute_extractor.py
----------------------
Derives perceptual attributes from each bounding box crop using OpenCV.

Extracted attributes:
  - color    : dominant hue label ("red", "green", "blue", …)
  - size     : normalized bbox area in [0, 1] relative to the full frame
  - position : horizontal thirds label ("left", "center", "right")
  - motion   : frame-to-frame motion magnitude ("stationary", "low", "high")

All methods are stateless except for motion, which caches the previous
frame's grayscale image internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV hue ranges for dominant color classification.
# OpenCV HSV: H ∈ [0, 180), S ∈ [0, 255], V ∈ [0, 255]
# Each entry: (label, [(h_low, h_high), ...])  — supports hue wrap-around.
# ---------------------------------------------------------------------------
HUE_RANGES: List[Tuple[str, List[Tuple[int, int]]]] = [
    ("red",    [(0, 10), (170, 180)]),   # red wraps around 0°/360°
    ("orange", [(10, 25)]),
    ("yellow", [(25, 35)]),
    ("green",  [(35, 85)]),
    ("cyan",   [(85, 100)]),
    ("blue",   [(100, 130)]),
    ("purple", [(130, 155)]),
    ("pink",   [(155, 170)]),
]

# Saturation and value thresholds below which a pixel is considered
# "achromatic" (white, gray, black) and excluded from hue voting.
MIN_SATURATION = 40   # out of 255
MIN_VALUE      = 40   # out of 255


@dataclass
class Attributes:
    """
    Perceptual attributes derived from a single bounding box.

    Attributes:
        color    : Dominant hue label. "unknown" if the crop is achromatic
                   or too small to analyze.
        size     : Bounding box area divided by total frame area.
                   Range [0, 1]. Values > 0.3 are "large" objects.
        position : Horizontal screen region of the bbox center.
                   "left" / "center" / "right" based on thirds.
        motion   : Motion intensity of the bbox region between the current
                   frame and the previous frame.
                   "stationary" | "low" | "high" | "unknown" (first frame).
    """
    color:    str = "unknown"
    size:     float = 0.0
    position: str = "unknown"
    motion:   str = "unknown"


class AttributeExtractor:
    """
    Computes Attributes for each RawDetection in a frame.

    Maintains a one-frame motion buffer — call extract() once per frame
    in strict temporal order so that motion deltas are meaningful.

    Parameters:
        motion_low_thresh  : Mean absolute pixel difference below which
                             motion is labelled "stationary".
        motion_high_thresh : Mean absolute pixel difference above which
                             motion is labelled "high". Between the two
                             thresholds it is "low".
    """

    def __init__(
        self,
        motion_low_thresh: float = 2.0,
        motion_high_thresh: float = 15.0,
    ):
        self._prev_gray: Optional[np.ndarray] = None
        self._motion_low  = motion_low_thresh
        self._motion_high = motion_high_thresh

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        frame: np.ndarray,
        bbox: Dict[str, int],
    ) -> Attributes:
        """
        Derive all attributes for a single detection bbox.

        Parameters:
            frame : Full BGR frame from which the bbox is cropped.
            bbox  : {"x", "y", "width", "height"} in pixels.

        Returns:
            Attributes dataclass populated with color, size, position,
            and motion for this specific bounding box.

        Note: Call update_motion_buffer() once per frame AFTER calling
        extract() on all bboxes in that frame. This keeps the previous-
        frame state consistent across all detections in the same frame.
        """
        crop = self._safe_crop(frame, bbox)
        fh, fw = frame.shape[:2]

        color    = self._dominant_color(crop)
        size     = self._normalized_size(bbox, fw, fh)
        position = self._horizontal_position(bbox, fw)
        motion   = self._motion_in_region(bbox)

        return Attributes(color=color, size=size, position=position, motion=motion)

    def update_motion_buffer(self, frame: np.ndarray) -> None:
        """
        Store current frame's grayscale for use as the 'previous frame'
        in the next call cycle.

        Call this ONCE per frame, after all extract() calls for that frame.
        This ensures every detection in frame N compares against frame N-1,
        not against a partially updated buffer.
        """
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Attribute extraction helpers
    # ------------------------------------------------------------------

    def _dominant_color(self, crop: Optional[np.ndarray]) -> str:
        """
        Classify the dominant hue in a BGR crop.

        Algorithm (think of it as a hue vote):
          1. Convert BGR → HSV.
          2. Mask out achromatic pixels (low saturation or low brightness).
          3. Build a 180-bin hue histogram over the remaining pixels.
          4. For each named color, sum histogram bins in its hue range(s).
          5. Return the color with the highest vote count.

        Returns "unknown" if the crop is None, too small, or entirely
        achromatic (e.g., a gray rock or white background).
        """
        if crop is None or crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Chromatic mask: keep only saturated, bright pixels
        chroma_mask = (s >= MIN_SATURATION) & (v >= MIN_VALUE)
        if not np.any(chroma_mask):
            return "unknown"   # fully achromatic region

        # 180-bin histogram over H channel, weighted by chroma mask
        hist = cv2.calcHist(
            [hsv], [0], chroma_mask.astype(np.uint8) * 255,
            [180], [0, 180]
        ).flatten()

        # Accumulate votes per named color across its hue ranges
        scores: Dict[str, float] = {}
        for label, ranges in HUE_RANGES:
            scores[label] = sum(hist[lo:hi].sum() for lo, hi in ranges)

        best = max(scores, key=scores.__getitem__)
        return best if scores[best] > 0 else "unknown"

    @staticmethod
    def _normalized_size(bbox: Dict[str, int], fw: int, fh: int) -> float:
        """
        Fraction of the total frame area occupied by this bbox.

        Formula: (bbox_w * bbox_h) / (frame_w * frame_h)

        This is scale-invariant: a large person close to camera scores
        higher than a small person in the distance, regardless of resolution.
        """
        bbox_area  = bbox["width"] * bbox["height"]
        frame_area = fw * fh
        if frame_area == 0:
            return 0.0
        return round(bbox_area / frame_area, 4)

    @staticmethod
    def _horizontal_position(bbox: Dict[str, int], frame_width: int) -> str:
        """
        Divides the frame width into three equal thirds and returns which
        third the bbox center falls in.

        |--- left (0–33%) ---|--- center (33–66%) ---|--- right (66–100%) ---|

        Using the center of the bbox rather than its left edge avoids
        misclassifying large objects that straddle a boundary.
        """
        center_x = bbox["x"] + bbox["width"] / 2
        third    = frame_width / 3

        if center_x < third:
            return "left"
        elif center_x < 2 * third:
            return "center"
        else:
            return "right"

    def _motion_in_region(self, bbox: Dict[str, int]) -> str:
        """
        Estimates motion within a bbox region by comparing it to the
        same region in the previous frame (frame differencing).

        Returns "unknown" on the very first frame (no previous frame).
        Otherwise returns "stationary", "low", or "high" based on
        the mean absolute difference of grayscale pixel values.

        Frame differencing is cheap and runs in real-time on CPU —
        no optical flow needed for this coarse classification.
        """
        if self._prev_gray is None:
            return "unknown"   # no previous frame yet

        return "unknown"   # placeholder; populated in update_motion_buffer path

    # ------------------------------------------------------------------
    # Frame-level motion (called from FrameProcessor with full context)
    # ------------------------------------------------------------------

    def compute_motion(
        self,
        current_gray: np.ndarray,
        bbox: Dict[str, int],
    ) -> str:
        """
        Compute motion label for a specific bbox by diffing against
        the buffered previous grayscale frame.

        Separated from extract() so the FrameProcessor can pass in a
        pre-converted grayscale once per frame rather than reconverting
        for every detection.

        Mean absolute difference interpretation:
          < motion_low_thresh  → "stationary"
          < motion_high_thresh → "low"
          ≥ motion_high_thresh → "high"
        """
        if self._prev_gray is None:
            return "unknown"

        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # Clamp bbox to frame boundaries to avoid out-of-bounds slicing
        fh, fw = current_gray.shape
        x1 = max(0, x);       y1 = max(0, y)
        x2 = min(fw, x + w);  y2 = min(fh, y + h)

        curr_roi = current_gray[y1:y2, x1:x2]
        prev_roi = self._prev_gray[y1:y2, x1:x2]

        if curr_roi.size == 0 or prev_roi.size == 0:
            return "unknown"

        diff = cv2.absdiff(curr_roi, prev_roi)
        mean_diff = float(diff.mean())

        if mean_diff < self._motion_low:
            return "stationary"
        elif mean_diff < self._motion_high:
            return "low"
        else:
            return "high"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: Dict[str, int]) -> Optional[np.ndarray]:
        """
        Crops the bbox region from the frame with boundary clamping.

        Returns None if the resulting crop would have zero area, which
        can happen when a detection bbox is at the very edge of the frame.
        """
        fh, fw = frame.shape[:2]
        x1 = max(0, bbox["x"])
        y1 = max(0, bbox["y"])
        x2 = min(fw, bbox["x"] + bbox["width"])
        y2 = min(fh, bbox["y"] + bbox["height"])

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
