"""
frame_processor.py
------------------
Orchestrates Detector + AttributeExtractor → FrameData.

This is the single entry point the rest of the pipeline interacts with.
It owns the per-frame loop logic and produces the canonical FrameData
objects that flow into the semantic alignment and scoring stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

from .detector import Detector, RawDetection
from .attribute_extractor import AttributeExtractor, Attributes
from .frame_source import FrameSource, RawFrame


# ---------------------------------------------------------------------------
# Output schema (mirrors the spec in the project overview doc)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Fully-enriched detection record for a single bounding box.

    Fields:
        label           : COCO class name from YOLO (e.g., "person").
        confidence      : YOLO detection confidence [0, 1].
        bbox            : {"x", "y", "width", "height"} in pixels.
        category        : Semantic category after normalization (e.g.,
                          "person" might become "child" after LLM mapping).
                          None until the semantic alignment stage fills it.
        attributes      : List of string attribute labels derived by
                          AttributeExtractor, e.g. ["red", "large", "center"].
        state           : Inferred behavioral/physical states, e.g.
                          ["stationary", "lying"] — populated downstream.
        relevance_score : Final score assigned by the scoring algorithm.
                          None until the scoring stage runs.
    """
    label:           str
    confidence:      float
    bbox:            dict
    category:        Optional[str]       = None
    attributes:      List[str]           = field(default_factory=list)
    state:           List[str]           = field(default_factory=list)
    relevance_score: Optional[float]    = None


@dataclass
class FrameData:
    """
    Complete processed output for a single video frame.

    Fields:
        frame_id   : Zero-based index of this frame in the stream.
        timestamp  : Seconds since stream start (or file mtime for images).
        detections : All Detection objects found in this frame, sorted by
                     confidence descending.
    """
    frame_id:   int
    timestamp:  float
    detections: List[Detection]


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class FrameProcessor:
    """
    Ties together Detector and AttributeExtractor into a single pipeline step.

    Typical usage:
        processor = FrameProcessor()
        for raw in FrameSource("video.mp4"):
            frame_data = processor.process(raw)
            scoring_stage.score(frame_data, target)

    Parameters:
        detector            : A configured Detector instance. If None, one
                              is created with defaults (yolov8n + openvino).
        attribute_extractor : A configured AttributeExtractor. If None, one
                              is created with defaults.
        infer_states        : If True, runs simple rule-based state inference
                              (posture, activity) on each detection before
                              returning FrameData. Adds negligible cost.
    """

    def __init__(
        self,
        detector: Optional[Detector] = None,
        attribute_extractor: Optional[AttributeExtractor] = None,
        infer_states: bool = True,
    ):
        self._detector   = detector   or Detector()
        self._extractor  = attribute_extractor or AttributeExtractor()
        self._infer_states = infer_states

        # Grayscale buffer shared across calls to avoid repeated conversion
        self._current_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, raw: RawFrame) -> FrameData:
        """
        Full pipeline for a single frame:
          1. Run YOLO detection on the raw image.
          2. Convert frame to grayscale once (reused for all bbox motion calc).
          3. For each detection, extract attributes (color, size, position,
             motion) via AttributeExtractor.
          4. Pack attributes into string list for the Detection record.
          5. Optionally infer behavioral states (posture, activity).
          6. Advance the motion buffer to this frame.
          7. Return a FrameData object.

        Parameters:
            raw : RawFrame from FrameSource containing frame_id, timestamp,
                  and the BGR image array.

        Returns:
            FrameData with all detections enriched.
        """
        frame = raw.image
        self._current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_detections: List[RawDetection] = self._detector.detect(frame)

        detections: List[Detection] = []
        for rd in raw_detections:
            attrs = self._extractor.extract(frame, rd.bbox)

            # Motion needs the current grayscale + buffered previous frame
            motion = self._extractor.compute_motion(self._current_gray, rd.bbox)
            attrs.motion = motion

            attr_strings = self._attrs_to_strings(attrs)

            det = Detection(
                label=rd.label,
                confidence=rd.confidence,
                bbox=rd.bbox,
                attributes=attr_strings,
            )

            if self._infer_states:
                det.state = self._infer_state(attrs)

            detections.append(det)

        # Sort highest-confidence detections first
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Advance motion buffer AFTER all detections are processed this frame
        self._extractor.update_motion_buffer(frame)

        return FrameData(
            frame_id=raw.frame_id,
            timestamp=raw.timestamp,
            detections=detections,
        )

    def process_stream(self, source: FrameSource):
        """
        Convenience generator: process every frame from a FrameSource.

        Yields FrameData objects in stream order.

        Usage:
            for fd in processor.process_stream(FrameSource(0)):
                print(fd)
        """
        for raw in source:
            yield self.process(raw)

    # ------------------------------------------------------------------
    # Attribute packing
    # ------------------------------------------------------------------

    @staticmethod
    def _attrs_to_strings(attrs: Attributes) -> List[str]:
        """
        Converts an Attributes dataclass into a flat list of string labels
        that can be used by the downstream semantic alignment and scoring
        stages without importing vision internals.

        Size is bucketed into human-readable categories:
            tiny   : < 1% of frame area   (distant or very small object)
            small  : 1–5%
            medium : 5–20%
            large  : > 20%

        Motion and position are passed through as-is if not "unknown".
        Color is always included (even "unknown") so the scorer can handle
        the absence of color information explicitly.
        """
        result = [f"color:{attrs.color}"]

        # Size bucket
        s = attrs.size
        if s < 0.01:
            result.append("size:tiny")
        elif s < 0.05:
            result.append("size:small")
        elif s < 0.20:
            result.append("size:medium")
        else:
            result.append("size:large")

        if attrs.position != "unknown":
            result.append(f"position:{attrs.position}")

        if attrs.motion != "unknown":
            result.append(f"motion:{attrs.motion}")

        return result

    # ------------------------------------------------------------------
    # State inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_state(attrs: Attributes) -> List[str]:
        """
        Rule-based inference of behavioral and physical states.

        This is intentionally simple and deterministic — it provides
        a first-pass signal that can be refined by the scoring algorithm
        or replaced by a more sophisticated model later.

        Current rules:
            motion == "stationary"  → "stationary"
            motion in {"low","high"} → "moving"
            size >= 0.20 AND motion == "stationary" → adds "possibly_lying"
                (large + not moving is a rough proxy for a prone person)
            size < 0.01             → "distant"

        "possibly_lying" is deliberately hedged — it is a weak signal
        and the scoring stage should weight it accordingly.
        """
        states: List[str] = []

        if attrs.motion == "stationary":
            states.append("stationary")
        elif attrs.motion in ("low", "high"):
            states.append("moving")

        if attrs.size >= 0.20 and attrs.motion == "stationary":
            states.append("possibly_lying")

        if attrs.size < 0.01:
            states.append("distant")

        return states
