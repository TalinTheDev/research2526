"""
visualizer.py
-------------
Optional debug visualizer that draws detection boxes, labels, attributes,
and states on each frame in a live OpenCV window.

Usage:
    viz = Visualizer()
    for fd in processor.process_stream(source):
        viz.show(frame, fd)          # pass the raw frame + FrameData
        if viz.should_quit():        # press Q or close window to stop
            break
    viz.close()
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from .frame_processor import FrameData, Detection


# Color palette for different YOLO labels (BGR)
_PALETTE = [
    (52,  152, 219),   # blue
    (46,  204, 113),   # green
    (231, 76,  60 ),   # red
    (155, 89,  182),   # purple
    (230, 126, 34 ),   # orange
    (26,  188, 156),   # teal
    (241, 196, 15 ),   # yellow
    (236, 240, 241),   # white
]

def _color_for(label: str) -> tuple:
    """Deterministic color per label string."""
    return _PALETTE[hash(label) % len(_PALETTE)]


class Visualizer:
    """
    Draws bounding boxes and detection metadata onto frames and displays
    them in a live OpenCV window.

    Parameters:
        window_name  : Title of the OpenCV window.
        scale        : Resize factor applied before display only —
                       does not affect the frame passed to the processor.
                       0.75 is useful for high-res webcams.
        show_attrs   : Draw attribute strings (color, size, position, motion).
        show_state   : Draw inferred state strings.
        show_conf    : Draw confidence score on each box.
        show_fps     : Show a live FPS counter in the top-left corner.
    """

    def __init__(
        self,
        window_name: str = "Seeker Vision Debug",
        scale: float = 1.0,
        show_attrs: bool = True,
        show_state: bool = True,
        show_conf: bool = True,
        show_fps: bool = True,
    ):
        self._window = window_name
        self._scale = scale
        self._show_attrs = show_attrs
        self._show_state = show_state
        self._show_conf = show_conf
        self._show_fps = show_fps

        self._prev_tick = cv2.getTickCount()
        self._fps = 0.0

        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def show(self, frame: np.ndarray, frame_data: FrameData) -> None:
        """
        Draw all detections from frame_data onto frame and display.

        Parameters:
            frame      : The original BGR frame (unmodified — we copy it).
            frame_data : FrameData produced by FrameProcessor.process().
        """
        canvas = frame.copy()

        for det in frame_data.detections:
            self._draw_detection(canvas, det)

        if self._show_fps:
            self._draw_fps(canvas)

        self._draw_frame_info(canvas, frame_data)

        if self._scale != 1.0:
            h, w = canvas.shape[:2]
            canvas = cv2.resize(canvas, (int(w * self._scale), int(h * self._scale)))

        cv2.imshow(self._window, canvas)
        cv2.waitKey(1)   # 1 ms pump — non-blocking

    def should_quit(self) -> bool:
        """
        Returns True if the user pressed Q/q or closed the window.
        Call once per frame after show().
        """
        key = cv2.waitKey(1) & 0xFF
        # Window closed check
        try:
            visible = cv2.getWindowProperty(self._window, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                return True
        except cv2.error:
            return True
        return key in (ord('q'), ord('Q'), 27)  # Q or Escape

    def close(self) -> None:
        """Destroy the OpenCV window."""
        cv2.destroyWindow(self._window)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_detection(self, canvas: np.ndarray, det: Detection) -> None:
        """
        Draws a single detection:
          - Filled semi-transparent bbox background
          - Colored border rectangle
          - Label + confidence on top bar
          - Attributes listed below the top bar
          - States listed after attributes
        """
        x, y, w, h = det.bbox["x"], det.bbox["y"], det.bbox["width"], det.bbox["height"]
        color = _color_for(det.label)

        # Semi-transparent fill inside the box
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # Solid border
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        # ── top label bar ────────────────────────────────────────────
        conf_str = f" {det.confidence:.2f}" if self._show_conf else ""
        header = f"{det.label}{conf_str}"
        (tw, th), baseline = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bar_y1 = max(0, y - th - baseline - 6)
        bar_y2 = y
        cv2.rectangle(canvas, (x, bar_y1), (x + tw + 6, bar_y2), color, -1)
        cv2.putText(canvas, header, (x + 3, y - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ── attribute + state lines inside the box ───────────────────
        lines = []
        if self._show_attrs and det.attributes:
            lines += det.attributes
        if self._show_state and det.state:
            lines += [f"[{s}]" for s in det.state]

        line_h = 16
        for i, line in enumerate(lines):
            ly = y + 14 + i * line_h
            if ly + line_h > y + h:   # don't overflow the box
                break
            # Small dark background pill for readability
            (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(canvas, (x + 2, ly - 12), (x + lw + 6, ly + 2), (20, 20, 20), -1)
            cv2.putText(canvas, line, (x + 4, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_fps(self, canvas: np.ndarray) -> None:
        """
        Computes and draws a rolling FPS counter using OpenCV tick frequency.
        Placed in the top-left corner with a dark background for contrast.
        """
        now = cv2.getTickCount()
        elapsed = (now - self._prev_tick) / cv2.getTickFrequency()
        if elapsed > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / elapsed)  # exponential smoothing
        self._prev_tick = now

        txt = f"FPS: {self._fps:.1f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (8, 8), (tw + 16, th + 16), (20, 20, 20), -1)
        cv2.putText(canvas, txt, (12, th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2, cv2.LINE_AA)

    def _draw_frame_info(self, canvas: np.ndarray, fd: FrameData) -> None:
        """
        Draws frame_id, timestamp, and detection count in the bottom-left corner.
        """
        h = canvas.shape[0]
        info = f"frame={fd.frame_id}  t={fd.timestamp:.2f}s  detections={len(fd.detections)}"
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (8, h - th - 16), (tw + 16, h - 6), (20, 20, 20), -1)
        cv2.putText(canvas, info, (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
