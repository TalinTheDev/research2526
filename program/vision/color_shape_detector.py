"""
color_shape_detector.py
-----------------------
OpenCV-based detector that replaces Detector for synthetic test frames.

Detects solid-colored shapes using HSV color masking + contour finding.
Outputs RawDetection objects identical to the real Detector so the rest
of the pipeline (AttributeExtractor, FrameProcessor, Visualizer) works
completely unchanged.

Color → label mapping is user-defined. Default:
    red    → person
    blue   → animal
    yellow → debris
    orange → landmark
    purple → vehicle
    cyan   → equipment
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detector import RawDetection


# ---------------------------------------------------------------------------
# HSV ranges per named color — same structure as attribute_extractor.py
# H ∈ [0, 180),  S ∈ [0, 255],  V ∈ [0, 255]
# Each entry: list of (h_low, h_high, s_low, v_low) tuples
# ---------------------------------------------------------------------------
COLOR_HSV_RANGES: Dict[str, List[Tuple[int, int, int, int]]] = {
    "red":    [(0, 10, 80, 60), (170, 180, 80, 60)],   # wraps around 0°
    "orange": [(10, 25, 100, 80)],
    "yellow": [(25, 35, 100, 80)],
    "green":  [(35, 85, 60, 40)],
    "cyan":   [(85, 100, 80, 60)],
    "blue":   [(100, 130, 80, 60)],
    "purple": [(130, 155, 60, 50)],
    "pink":   [(155, 170, 60, 60)],
}

# Default color → semantic label mapping
DEFAULT_COLOR_LABEL_MAP: Dict[str, str] = {
    "red":    "person",
    "blue":   "animal",
    "yellow": "debris",
    "orange": "landmark",
    "purple": "vehicle",
    "cyan":   "equipment",
}


class ColorShapeDetector:
    """
    Drop-in replacement for Detector that uses OpenCV HSV masking
    and contour detection instead of YOLO.

    Works on any frame containing solid-colored shapes on a contrasting
    background. Designed for synthetic test frames from TestFrameGenerator
    but will work on any sufficiently high-contrast colored objects.

    Parameters:
        color_label_map   : Dict mapping color name → label string.
                            Defaults to DEFAULT_COLOR_LABEL_MAP.
        min_area          : Minimum contour area in pixels. Filters out
                            noise and tiny artifacts.
        max_area_ratio    : Maximum bbox area as fraction of frame area.
                            Filters out detections that cover the whole frame
                            (e.g., background color bleed).
        confidence_thresh : Detections with computed confidence below this
                            are discarded. Confidence is derived from how
                            cleanly the contour matches its color mask.
        morph_kernel_size : Size of morphological kernel used to clean up
                            color masks before contour finding. Larger values
                            fill more holes but blur boundaries.
    """

    def __init__(
        self,
        color_label_map: Optional[Dict[str, str]] = None,
        min_area: int = 400,
        max_area_ratio: float = 0.70,
        confidence_thresh: float = 0.40,
        morph_kernel_size: int = 5,
    ):
        self._map = color_label_map or DEFAULT_COLOR_LABEL_MAP.copy()
        self._min_area = min_area
        self._max_area_ratio = max_area_ratio
        self._conf_thresh = confidence_thresh
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

        # Assign a unique integer class_id per label for RawDetection
        labels = list(dict.fromkeys(self._map.values()))  # preserve order, dedupe
        self._label_ids: Dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}

    # ------------------------------------------------------------------
    # Public interface — matches Detector.detect() signature exactly
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Run color-shape detection on a single BGR frame.

        Parameters:
            frame : H×W×3 BGR numpy array.

        Returns:
            List of RawDetection objects, one per detected colored shape,
            filtered by min_area, max_area_ratio, and confidence_thresh.
        """
        fh, fw = frame.shape[:2]
        frame_area = fh * fw
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detections: List[RawDetection] = []

        for color_name, ranges in COLOR_HSV_RANGES.items():
            if color_name not in self._map:
                continue  # not in user's label map, skip

            label = self._map[color_name]
            mask = self._build_mask(hsv, ranges)
            contours = self._find_contours(mask)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self._min_area:
                    continue
                if area / frame_area > self._max_area_ratio:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                confidence = self._compute_confidence(mask, x, y, w, h, area)

                if confidence < self._conf_thresh:
                    continue

                detections.append(RawDetection(
                    label=label,
                    confidence=round(confidence, 3),
                    bbox={"x": x, "y": y, "width": w, "height": h},
                    class_id=self._label_ids.get(label, 0),
                ))

        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_mask(
        self,
        hsv: np.ndarray,
        ranges: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Builds a binary mask for a color by OR-ing all its HSV ranges.

        Each range is (h_low, h_high, s_low, v_low). S and V upper bounds
        are always 255 — we only threshold the lower end to catch all
        saturations and brightnesses above the noise floor.

        Morphological close (dilate then erode) fills small holes inside
        shapes caused by specular highlights or aliasing.
        """
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (h_lo, h_hi, s_lo, v_lo) in ranges:
            lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            hi = np.array([h_hi, 255,  255 ], dtype=np.uint8)
            combined |= cv2.inRange(hsv, lo, hi)

        # Close small gaps, then open to remove isolated noise pixels
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self._kernel)
        return combined

    @staticmethod
    def _find_contours(mask: np.ndarray) -> List[np.ndarray]:
        """
        Finds external contours in the binary mask.

        RETR_EXTERNAL: only outermost contours (ignores holes inside shapes).
        CHAIN_APPROX_SIMPLE: compresses horizontal/vertical/diagonal segments
        to their endpoints — saves memory and speeds up boundingRect.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    @staticmethod
    def _compute_confidence(
        mask: np.ndarray,
        x: int, y: int, w: int, h: int,
        contour_area: float,
    ) -> float:
        """
        Estimates detection confidence as the ratio of actual contour area
        to bounding box area — think of it as "how filled is the bbox?".

        A perfect solid rectangle scores ~1.0.
        A circle scores ~0.785 (pi/4).
        A triangle scores ~0.5.
        Noise/artifacts score very low.

        This gives downstream stages a meaningful confidence signal even
        without an ML model producing probability scores.
        """
        bbox_area = w * h
        if bbox_area == 0:
            return 0.0
        fill_ratio = contour_area / bbox_area

        # Also factor in mask density within the bbox region
        roi = mask[y:y+h, x:x+w]
        mask_density = np.count_nonzero(roi) / bbox_area if bbox_area > 0 else 0.0

        # Weighted combination: fill_ratio is geometric, mask_density is photometric
        confidence = 0.6 * fill_ratio + 0.4 * mask_density
        return float(np.clip(confidence, 0.0, 1.0))
