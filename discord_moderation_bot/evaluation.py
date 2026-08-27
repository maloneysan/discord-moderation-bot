from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .service import ModerationService


VALID_CATEGORIES = frozenset(
    {"discrimination", "cynicism", "sexual_content", "sensitive_term", "drug_content"}
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    dimension: str
    text: str
    expected_categories: frozenset[str]
    reply_context: Optional[str] = None
    recent_context: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    case_count: int
    exact_matches: int
    true_positives: int
    false_positives: int
    false_negatives: int
    mismatched_case_ids: Tuple[str, ...]

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.case_count if self.case_count else 0.0

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else 1.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 1.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def load_evaluation_cases(path: Path) -> Tuple[EvaluationCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation dataset must contain a non-empty cases list")

    cases = []
    seen_ids = set()
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation case must be an object")
        case_id = raw.get("id")
        dimension = raw.get("dimension")
        text = raw.get("text")
        expected = raw.get("expected_categories")
        reply_context = raw.get("reply_context")
        recent_context = raw.get("recent_context", [])
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("evaluation case id must be unique and non-empty")
        if not isinstance(dimension, str) or not dimension:
            raise ValueError(f"evaluation case {case_id} has no dimension")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"evaluation case {case_id} has no text")
        if not isinstance(expected, list) or not all(
            isinstance(item, str) and item in VALID_CATEGORIES for item in expected
        ):
            raise ValueError(f"evaluation case {case_id} has invalid categories")
        if reply_context is not None and not isinstance(reply_context, str):
            raise ValueError(f"evaluation case {case_id} has invalid reply context")
        if not isinstance(recent_context, list) or not all(
            isinstance(item, str) for item in recent_context
        ):
            raise ValueError(f"evaluation case {case_id} has invalid recent context")
        seen_ids.add(case_id)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                dimension=dimension,
                text=text,
                expected_categories=frozenset(expected),
                reply_context=reply_context,
                recent_context=tuple(recent_context),
            )
        )
    return tuple(cases)


async def evaluate_cases(
    service: ModerationService,
    cases: Iterable[EvaluationCase],
) -> EvaluationReport:
    case_list = tuple(cases)
    exact_matches = true_positives = false_positives = false_negatives = 0
    mismatches = []
    for case in case_list:
        result = await service.analyze(
            case.text,
            reply_context=case.reply_context,
            recent_context=case.recent_context,
        )
        actual = frozenset(item.category for item in result.detections)
        if actual == case.expected_categories:
            exact_matches += 1
        else:
            mismatches.append(case.case_id)
        true_positives += len(actual & case.expected_categories)
        false_positives += len(actual - case.expected_categories)
        false_negatives += len(case.expected_categories - actual)
    return EvaluationReport(
        case_count=len(case_list),
        exact_matches=exact_matches,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        mismatched_case_ids=tuple(mismatches),
    )
