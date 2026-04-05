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
from .color_shape_detector import ColorShapeDetector, DEFAULT_COLOR_LABEL_MAP
from .test_frame_generator import TestFrameGenerator, ShapeSpec

__all__ = [
    "FrameSource", "RawFrame", "SourceType",
    "Detector", "RawDetection",
    "AttributeExtractor", "Attributes",
    "FrameProcessor", "FrameData", "Detection",
    "Visualizer",
    "ColorShapeDetector", "DEFAULT_COLOR_LABEL_MAP",
    "TestFrameGenerator", "ShapeSpec",
]
