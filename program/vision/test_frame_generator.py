"""
test_frame_generator.py
-----------------------
Generates synthetic test frames: colored solid shapes on a procedurally
generated forest-green background.

Each shape has a defined color, geometry, position, and optional motion.
The generator produces a FrameSource-compatible iterator so it plugs
directly into FrameProcessor.process_stream().

Quick start:
    gen = TestFrameGenerator()
    gen.add_shape(ShapeSpec("red",    "rectangle", (200, 150), (80, 120)))  # person
    gen.add_shape(ShapeSpec("blue",   "circle",    (400, 250), (40, 40)))   # animal
    gen.add_shape(ShapeSpec("yellow", "triangle",  (550, 300), (60, 60)))   # debris

    for raw in gen.stream(num_frames=60):
        fd = processor.process(raw)
        viz.show(raw.image, fd)
"""

from __future__ import annotations

import cv2
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

from .frame_source import RawFrame


# ---------------------------------------------------------------------------
# ShapeSpec — defines a single object in the scene
# ---------------------------------------------------------------------------

@dataclass
class ShapeSpec:
    """
    Describes a single colored shape to draw on the test frame.

    Parameters:
        color     : Color name matching COLOR_HSV_RANGES in ColorShapeDetector.
                    e.g. "red", "blue", "yellow", "orange", "purple", "cyan".
        shape     : Geometry type: "rectangle", "circle", or "triangle".
        position  : (x, y) center of the shape in pixels.
        size      : (width, height) of the bounding box.
                    For circles, width is used as diameter.
        velocity  : (vx, vy) pixels per frame. (0, 0) = stationary.
        label     : Optional human-readable tag drawn above the shape.
                    If None, the color name is used.
        opacity   : Blend factor [0, 1]. 1.0 = fully opaque.
    """
    color:    str
    shape:    str                          # "rectangle" | "circle" | "triangle"
    position: Tuple[int, int]             # (cx, cy)
    size:     Tuple[int, int]             # (w, h)
    velocity: Tuple[float, float] = (0.0, 0.0)
    label:    Optional[str]       = None
    opacity:  float               = 1.0

    def __post_init__(self):
        # Store position as floats internally for sub-pixel motion tracking
        self._x = float(self.position[0])
        self._y = float(self.position[1])

    def step(self, frame_w: int, frame_h: int) -> None:
        """
        Advance position by velocity. Bounces off frame edges so shapes
        stay visible indefinitely.
        """
        hw = self.size[0] / 2
        hh = self.size[1] / 2

        self._x += self.velocity[0]
        self._y += self.velocity[1]

        # Bounce horizontally
        if self._x - hw < 0:
            self._x = hw
            self.velocity = (-self.velocity[0], self.velocity[1])
        elif self._x + hw > frame_w:
            self._x = frame_w - hw
            self.velocity = (-self.velocity[0], self.velocity[1])

        # Bounce vertically
        if self._y - hh < 0:
            self._y = hh
            self.velocity = (self.velocity[0], -self.velocity[1])
        elif self._y + hh > frame_h:
            self._y = frame_h - hh
            self.velocity = (self.velocity[0], -self.velocity[1])

        self.position = (int(self._x), int(self._y))


# ---------------------------------------------------------------------------
# BGR values for each color name
# ---------------------------------------------------------------------------
COLOR_BGR: dict = {
    "red":    (  0,   0, 220),
    "orange": (  0, 120, 230),
    "yellow": (  0, 220, 220),
    "green":  (  0, 180,  40),
    "cyan":   (200, 200,   0),
    "blue":   (220,  60,   0),
    "purple": (180,  40, 160),
    "pink":   (180,  80, 220),
}


# ---------------------------------------------------------------------------
# TestFrameGenerator
# ---------------------------------------------------------------------------

class TestFrameGenerator:
    """
    Generates synthetic BGR frames with colored shapes on a forest background.

    The background is a procedurally textured green gradient with noise to
    loosely resemble foliage. Shapes are drawn on top with optional labels.

    Parameters:
        width, height  : Frame dimensions in pixels.
        fps            : Simulated FPS — used for timestamp calculation only.
        bg_seed        : Random seed for background texture. Same seed always
                         produces the same background.
    """

    def __init__(
        self,
        width: int  = 640,
        height: int = 480,
        fps: float  = 30.0,
        bg_seed: int = 42,
    ):
        self._w = width
        self._h = height
        self._fps = fps
        self._shapes: List[ShapeSpec] = []
        self._background = self._make_background(width, height, bg_seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_shape(self, spec: ShapeSpec) -> "TestFrameGenerator":
        """
        Add a shape to the scene. Returns self for chaining.

            gen.add_shape(...).add_shape(...).add_shape(...)
        """
        self._shapes.append(spec)
        return self

    def stream(self, num_frames: int = 0) -> Generator[RawFrame, None, None]:
        """
        Yields RawFrame objects. Plug directly into FrameProcessor.process().

        Parameters:
            num_frames : Number of frames to generate. 0 = infinite.

        Usage:
            for raw in gen.stream(num_frames=120):
                fd = processor.process(raw)
        """
        frame_id = 0
        start = time.time()

        while True:
            if num_frames > 0 and frame_id >= num_frames:
                break

            frame = self._render()
            yield RawFrame(
                frame_id=frame_id,
                timestamp=frame_id / self._fps,
                image=frame,
            )

            # Advance all moving shapes
            for shape in self._shapes:
                shape.step(self._w, self._h)

            frame_id += 1

    def render_single(self) -> np.ndarray:
        """
        Render and return one frame without advancing the generator state.
        Useful for previewing the scene setup.
        """
        return self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> np.ndarray:
        """
        Compose background + all shapes into a single BGR frame.
        Shapes are drawn in list order (later shapes appear on top).
        """
        canvas = self._background.copy()
        for shape in self._shapes:
            self._draw_shape(canvas, shape)
        return canvas

    def _draw_shape(self, canvas: np.ndarray, spec: ShapeSpec) -> None:
        """
        Dispatches to the appropriate draw function based on spec.shape,
        then draws an optional label tag above the shape.
        """
        bgr = COLOR_BGR.get(spec.color, (200, 200, 200))
        cx, cy = spec.position
        w, h   = spec.size

        if spec.opacity < 1.0:
            overlay = canvas.copy()
            self._draw_geometry(overlay, spec.shape, cx, cy, w, h, bgr)
            cv2.addWeighted(overlay, spec.opacity, canvas, 1 - spec.opacity, 0, canvas)
        else:
            self._draw_geometry(canvas, spec.shape, cx, cy, w, h, bgr)

        # Label tag above the shape
        tag = spec.label or spec.color
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = cx - tw // 2
        ty = cy - h // 2 - 6
        cv2.rectangle(canvas, (tx - 3, ty - th - 2), (tx + tw + 3, ty + 2), (20, 20, 20), -1)
        cv2.putText(canvas, tag, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_geometry(
        canvas: np.ndarray,
        shape: str,
        cx: int, cy: int,
        w: int, h: int,
        bgr: Tuple[int, int, int],
    ) -> None:
        """
        Draws a filled shape centered at (cx, cy).

        rectangle : standard axis-aligned filled rect.
        circle    : filled ellipse fitting inside (w, h).
        triangle  : filled isoceles triangle pointing up.
        """
        if shape == "rectangle":
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, -1)
            # Thin dark outline for crispness
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

        elif shape == "circle":
            cv2.ellipse(canvas, (cx, cy), (w // 2, h // 2), 0, 0, 360, bgr, -1)
            cv2.ellipse(canvas, (cx, cy), (w // 2, h // 2), 0, 0, 360, (20, 20, 20), 1)

        elif shape == "triangle":
            pts = np.array([
                [cx,         cy - h // 2],   # apex
                [cx - w // 2, cy + h // 2],  # bottom-left
                [cx + w // 2, cy + h // 2],  # bottom-right
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], bgr)
            cv2.polylines(canvas, [pts], isClosed=True, color=(20, 20, 20), thickness=1)

        else:
            # Fallback: draw a filled circle for unknown shape types
            cv2.circle(canvas, (cx, cy), min(w, h) // 2, bgr, -1)

    # ------------------------------------------------------------------
    # Background generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_background(w: int, h: int, seed: int) -> np.ndarray:
        """
        Creates a procedural forest-green background.

        Layers:
          1. Base dark-green gradient (darker at top, lighter toward bottom
             to simulate ground vs canopy lighting).
          2. Perlin-like noise via stacked OpenCV GaussianBlurs on random
             noise — cheap approximation that looks organic.
          3. Slight vignette to focus attention on the center.

        The result is a plausible abstract forest floor texture that gives
        the color detector enough contrast to isolate bright shapes.
        """
        rng = np.random.default_rng(seed)

        # 1. Green gradient base — H shifts slightly top to bottom
        base = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / h   # 0 at top, 1 at bottom
            # BGR: darker canopy green → warmer forest floor brown-green
            b = int(20  + t * 10)
            g = int(60  + t * 40)
            r = int(15  + t * 25)
            base[y, :] = (b, g, r)

        # 2. Layered noise for foliage texture
        noise = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
        for sigma in [3, 7, 15]:
            layer = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
            blurred = cv2.GaussianBlur(layer.astype(np.float32), (0, 0), sigma)
            noise = np.clip(noise.astype(np.int16) + blurred.astype(np.int16), 0, 255).astype(np.uint8)

        background = np.clip(base.astype(np.int16) + noise.astype(np.int16) // 3, 0, 255).astype(np.uint8)

        # 3. Vignette
        vy = np.linspace(0.75, 1.0, h)[:, None]
        vx = np.linspace(0.75, 1.0, w)[None, :]
        vignette = (vy * vx).astype(np.float32)
        vignette = cv2.GaussianBlur(vignette, (0, 0), w // 4)
        vignette = np.clip(vignette, 0.5, 1.0)[:, :, None]
        background = np.clip(background.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

        return background
