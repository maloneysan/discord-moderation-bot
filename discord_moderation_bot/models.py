from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CategoryDetection:
    """One category result. Message content is intentionally not retained."""

    category: str
    label: str
    score: int
    rule_ids: Tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class DetectionResult:
    """Immutable moderation result without source text."""

    detected: bool
    detections: Tuple[CategoryDetection, ...]

    @classmethod
    def empty(cls) -> "DetectionResult":
        return cls(detected=False, detections=())
