"""
frame_source.py
---------------
Unified iterator abstraction over all three input types:
webcam, video file, and image sequence.

Yields (frame_id, timestamp, np.ndarray) tuples regardless of source.
"""

import cv2
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, Optional
import numpy as np


class SourceType(Enum):
    WEBCAM = auto()
    VIDEO = auto()
    IMAGE_SEQUENCE = auto()


@dataclass
class RawFrame:
    """
    A single raw frame from any source.

    Attributes:
        frame_id  : Zero-based index of this frame in the stream.
        timestamp : Seconds since stream start (or file mtime for images).
        image     : BGR numpy array as returned by OpenCV.
    """
    frame_id: int
    timestamp: float
    image: np.ndarray


class FrameSource:
    """
    Wraps webcam, video file, or image sequence behind a single iterator.

    Parameters:
        source      : int (webcam index), str (video/image path), or
                      list[str] (ordered image paths).
        target_fps  : For webcam only — attempts to set capture FPS.
                      Has no effect on video files or image sequences.
        loop        : If True, video files restart after the last frame.
                      Has no effect on webcam or image sequences.

    Usage:
        for raw in FrameSource(0):                     # webcam
            process(raw)

        for raw in FrameSource("clip.mp4"):            # video
            process(raw)

        for raw in FrameSource(["a.jpg", "b.jpg"]):   # image sequence
            process(raw)
    """

    def __init__(
        self,
        source: int | str | List[str],
        target_fps: int = 30,
        loop: bool = False,
    ):
        self._source = source
        self._target_fps = target_fps
        self._loop = loop
        self._type = self._detect_type(source)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def stream(self) -> Generator[RawFrame, None, None]:
        """
        Generator that yields RawFrame objects one at a time.

        For webcam: runs indefinitely until KeyboardInterrupt or cap.read() fails.
        For video:  stops at last frame, or loops if self._loop is True.
        For images: yields each image once in order, then stops.
        """
        if self._type == SourceType.WEBCAM:
            yield from self._stream_webcam()
        elif self._type == SourceType.VIDEO:
            yield from self._stream_video()
        else:
            yield from self._stream_images()

    def __iter__(self):
        return self.stream()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_type(source) -> SourceType:
        """
        Infers source type from the argument:
          - int              → webcam device index
          - list             → pre-built image path list
          - str ending in image ext  → single image (treated as 1-image sequence)
          - str (other)      → video file
        """
        if isinstance(source, int):
            return SourceType.WEBCAM
        if isinstance(source, list):
            return SourceType.IMAGE_SEQUENCE
        # single string: check extension
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTS:
            return SourceType.IMAGE_SEQUENCE
        return SourceType.VIDEO

    def _stream_webcam(self) -> Generator[RawFrame, None, None]:
        """
        Opens the webcam device, sets FPS, and yields frames.

        frame_id  : increments each captured frame.
        timestamp : wall-clock seconds via time.time().
        """
        cap = cv2.VideoCapture(self._source)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        frame_id = 0
        start_time = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield RawFrame(
                    frame_id=frame_id,
                    timestamp=time.time() - start_time,
                    image=frame,
                )
                frame_id += 1
        finally:
            cap.release()

    def _stream_video(self) -> Generator[RawFrame, None, None]:
        """
        Opens a video file and yields frames.

        frame_id  : absolute frame index (resets if loop=True).
        timestamp : derived from the video's own PTS in milliseconds
                    (CAP_PROP_POS_MSEC), converted to seconds.
        """
        frame_id = 0

        while True:  # outer loop handles looping
            cap = cv2.VideoCapture(self._source)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield RawFrame(frame_id=frame_id, timestamp=timestamp, image=frame)
                frame_id += 1

            cap.release()
            if not self._loop:
                break

    def _stream_images(self) -> Generator[RawFrame, None, None]:
        """
        Reads images from a list of file paths.

        If self._source is a single string it is wrapped in a list first.

        frame_id  : index into the list.
        timestamp : file modification time (mtime) in epoch seconds,
                    so images retain their real-world capture ordering.
        """
        paths = (
            [self._source]
            if isinstance(self._source, str)
            else self._source
        )

        for frame_id, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                # Skip unreadable files rather than crashing the pipeline.
                continue
            timestamp = os.path.getmtime(path)
            yield RawFrame(frame_id=frame_id, timestamp=timestamp, image=img)
