"""
scoring_interface.py
--------------------
Defines the ScoringAlgorithm protocol — the only contract between the
simulation and any scoring algorithm.

The simulation knows nothing about how scores are computed. It just calls
algo.score(detections, target), reads relevance_score off each Detection,
and steers toward the highest.

Plug in any algorithm by implementing ScoringAlgorithm:
    - ConfidenceOnlyScorer   (baseline — ignores target entirely)
    - ColorLabelScorer       (simple matching baseline)
    - YourAlgorithmV1        (slot in here)
    - YourAlgorithmV2        (slot in here)
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable
from vision.frame_processor import Detection


# ---------------------------------------------------------------------------
# Protocol — the only interface the simulation cares about
# ---------------------------------------------------------------------------

@runtime_checkable
class ScoringAlgorithm(Protocol):
    """
    Any object with a score() method matching this signature is a valid
    algorithm. No inheritance required — just implement the method.

    Parameters:
        detections : List of Detection objects from FrameProcessor.
                     Each has label, confidence, bbox, attributes, state.
                     relevance_score is None on entry — fill it in.
        target     : Dict describing what the robot is looking for.
                     Structure mirrors the LLM output schema:
                     {
                       "object":       str | None,
                       "state":        list[str],
                       "descriptions": list[str],   # e.g. ["red jacket"]
                     }

    Returns:
        The same detections list with relevance_score filled in on each.
        Scores should be in [0, 1] but the simulation only needs them to
        be comparable (higher = more relevant) — absolute scale doesn't matter.
    """
    def score(
        self,
        detections: List[Detection],
        target: dict,
    ) -> List[Detection]:
        ...

    @property
    def name(self) -> str:
        """Human-readable name shown in simulation output."""
        ...


# ---------------------------------------------------------------------------
# Baseline A — confidence only (ignores target description entirely)
# ---------------------------------------------------------------------------

class ConfidenceOnlyScorer:
    """
    Baseline scorer that uses YOLO/detector confidence as relevance score.
    Ignores the target description completely.

    Useful as a lower-bound comparison: a good semantic algorithm should
    outperform this by navigating toward the described target rather than
    just the most confidently detected object.
    """

    name = "ConfidenceOnly"

    def score(
        self,
        detections: List[Detection],
        target: dict,
    ) -> List[Detection]:
        for det in detections:
            det.relevance_score = det.confidence
        return detections


# ---------------------------------------------------------------------------
# Baseline B — simple color + label matching
# ---------------------------------------------------------------------------

class ColorLabelScorer:
    """
    Simple scorer that matches detections against the target description
    using color and label string matching. No semantic understanding.

    Scoring breakdown (all components in [0, 1], weighted sum):
        label match   : 0.5  — does det.label match target["object"]?
        color match   : 0.3  — does any det.attribute color match descriptions?
        confidence    : 0.2  — raw detector confidence as tiebreaker

    This is a reasonable "dumb" baseline that knows about the target but
    does no semantic reasoning.
    """

    name = "ColorLabel"

    def score(
        self,
        detections: List[Detection],
        target: dict,
    ) -> List[Detection]:
        target_label  = (target.get("object") or "").lower()
        target_descs  = [d.lower() for d in target.get("descriptions", [])]
        target_colors = [d for d in target_descs if d in self._known_colors()]

        for det in detections:
            label_score  = self._label_match(det.label, target_label)
            color_score  = self._color_match(det.attributes, target_colors)
            conf_score   = det.confidence

            det.relevance_score = round(
                0.5 * label_score +
                0.3 * color_score +
                0.2 * conf_score,
                4,
            )

        return detections

    # ------------------------------------------------------------------

    @staticmethod
    def _label_match(det_label: str, target_label: str) -> float:
        """1.0 if exact match, 0.5 if one contains the other, 0.0 otherwise."""
        if not target_label:
            return 0.5   # no target label specified — neutral
        dl = det_label.lower()
        if dl == target_label:
            return 1.0
        if dl in target_label or target_label in dl:
            return 0.5
        return 0.0

    @staticmethod
    def _color_match(attributes: List[str], target_colors: List[str]) -> float:
        """1.0 if any target color appears in attributes, 0.0 otherwise."""
        if not target_colors:
            return 0.5   # no color specified — neutral
        attr_colors = {
            a.split(":")[1] for a in attributes
            if a.startswith("color:") and a.split(":")[1] != "unknown"
        }
        return 1.0 if attr_colors & set(target_colors) else 0.0

    @staticmethod
    def _known_colors() -> set:
        return {"red","orange","yellow","green","cyan","blue","purple","pink","brown","white","black","gray"}
