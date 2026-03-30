"""
vision/
-------
Computer vision module for the seeker robot pipeline.

Public API:
    FrameSource      — unified input iterator (webcam / video / images)
    Detector         — YOLOv8 wrapper returning RawDetection list
    AttributeExtractor — OpenCV attribute derivation (color, size, position, motion)
    FrameProcessor   — orchestrator producing FrameData
    FrameData        — canonical output schema
    Detection        — single detection record
"""

from .frame_source import FrameSource, RawFrame, SourceType
from .detector import Detector, RawDetection
from .attribute_extractor import AttributeExtractor, Attributes
from .frame_processor import FrameProcessor, FrameData, Detection
from .visualizer import Visualizer

__all__ = [
    "FrameSource", "RawFrame", "SourceType",
    "Detector", "RawDetection",
    "AttributeExtractor", "Attributes",
    "FrameProcessor", "FrameData", "Detection",
    "Visualizer",
]
