"""
detector.py
-----------
YOLOv8n/s wrapper that runs inference on a single frame and returns
a list of raw detections structured for the downstream pipeline.

Targets Intel Arc iGPU via the OpenVINO backend bundled with Ultralytics.
Falls back to CPU if OpenVINO is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class RawDetection:
    """
    Raw output from YOLO for a single bounding box.

    Attributes:
        label       : COCO class name as a string, e.g. "person", "dog".
        confidence  : Detection confidence in [0, 1].
        bbox        : Dict with keys x, y, width, height — all in pixels,
                      relative to the original frame (not the YOLO input size).
                      x, y are the top-left corner.
        class_id    : Integer COCO class index, useful for fast lookup tables.
    """
    label: str
    confidence: float
    bbox: dict          # {"x": int, "y": int, "width": int, "height": int}
    class_id: int


class Detector:
    """
    Thin wrapper around a YOLOv8 model.

    At construction time the model is loaded once and kept in memory.
    Subsequent calls to detect() reuse the same model instance — this is
    important because YOLOv8 model loading is expensive (~0.5–2 s).

    Parameters:
        model_path        : Path to a .pt file or an already-exported
                            OpenVINO directory (recommended for Arc).
                            Defaults to "yolov8n.pt" (auto-downloaded on
                            first run by Ultralytics).
        confidence_thresh : Detections below this score are discarded.
                            0.35 is a good starting point for SAR use.
        device            : "openvino" for Intel Arc (fastest on iGPU),
                            "cpu" as fallback, "cuda" for Nvidia.
        classes           : Optional list of COCO class IDs to restrict
                            detection to. None means detect all 80 classes.

    Notes on OpenVINO export (one-time setup):
        from ultralytics import YOLO
        YOLO("yolov8n.pt").export(format="openvino")
        # produces yolov8n_openvino_model/ directory
        # pass that directory path as model_path
    """

    # COCO class IDs most relevant to search-and-rescue environments.
    # Feel free to extend or replace with a custom-trained model's labels.
    SAR_RELEVANT_CLASSES = {
        0:  "person",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        # Non-living but spatially significant:
        56: "chair",   # debris proxy
        57: "couch",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_thresh: float = 0.35,
        device: str = "openvino",
        classes: Optional[List[int]] = None,
    ):
        from ultralytics import YOLO  # deferred import — heavy dependency

        self._conf = confidence_thresh
        self._classes = classes
        self._device = device

        # Load model once. Ultralytics auto-downloads .pt files on first use.
        self._model = YOLO(model_path)

        print(f"[Detector] Loaded model '{model_path}' | device={device} | conf≥{confidence_thresh}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Run inference on a single BGR frame.

        Parameters:
            frame : H×W×3 numpy array (BGR, as returned by OpenCV).

        Returns:
            List of RawDetection objects, one per bounding box that
            survived the confidence threshold filter.

        Implementation notes:
            - verbose=False suppresses Ultralytics per-frame console output.
            - results[0] because we pass a single frame (batch size 1).
            - .boxes.data is a tensor of shape [N, 6]:
              columns → [x1, y1, x2, y2, confidence, class_id]
        """
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )

        return self._parse_results(results[0])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_results(self, result) -> List[RawDetection]:
        """
        Converts a single Ultralytics Results object into RawDetection list.

        Ultralytics stores boxes in xyxy format (top-left + bottom-right).
        We convert to xywh (top-left + size) for consistency with the
        FrameData schema and to make bbox area calculation trivial.
        """
        detections: List[RawDetection] = []

        if result.boxes is None:
            return detections

        names = result.names  # dict: {class_id: label_str}

        for box in result.boxes:
            # box.xyxy → tensor [[x1, y1, x2, y2]]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names.get(cls_id, f"class_{cls_id}")

            detections.append(RawDetection(
                label=label,
                confidence=conf,
                bbox={
                    "x":      int(x1),
                    "y":      int(y1),
                    "width":  int(x2 - x1),
                    "height": int(y2 - y1),
                },
                class_id=cls_id,
            ))

        return detections
