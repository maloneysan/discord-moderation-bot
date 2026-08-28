from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Pattern, Sequence, Tuple

from .models import CategoryDetection, DetectionResult


_CONTROL_CHARACTERS = re.compile(
    "[\u034f\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]"
)
_VARIATION_SELECTORS = re.compile("[\ufe00-\ufe0f\U000e0100-\U000e01ef]")
_WHITESPACE = re.compile(r"\s+")
_OBFUSCATION_SEPARATORS = re.compile(r"[._･・·•]+")


_LOCAL_REASONS = {
    "discrimination": (
        "属性や立場を理由に、相手を排除・劣等視・固定観念で扱う内容です。"
    ),
    "cynicism": "相手を見下したり、嘲笑・突き放しとして受け取られる内容です。",
    "sexual_content": "性的な話題や下ネタとして扱われる内容です。",
    "sensitive_term": "サーバーで指定された要注意語への言及です。",
    "drug_content": "違法薬物や薬物乱用に関連する内容です。",
}

_LOCAL_RULE_REASONS = {
    "discrimination.explicit_slur": "属性や障害などを蔑称で呼び、相手の尊厳を傷つけています。",
    "discrimination.exclusion_or_inferiority": "属性や立場を理由に、能力を低く扱ったり排除を求めています。",
    "discrimination.harmful_stereotype": "集団全体を否定的な固定観念で決めつけています。",
    "discrimination.dehumanization": "人を害虫や汚物などに例え、人間としての尊厳を否定しています。",
    "discrimination.rights_denial": "属性や立場を理由に、権利や参加機会を奪うことを求めています。",
    "discrimination.gender_role_denial": "性別役割を押し付け、相手の能力や行動を否定しています。",
    "discrimination.ableist_denial": "障害や疾病を甘え・怠けと決めつけ、本人の困難を否定しています。",
    "discrimination.identity_pathologizing": "性自認や性的指向を病気・異常として扱っています。",
    "cynicism.gendered_mockery": "性別規範を使って、相手を弱い・情けないと辱めています。",
    "cynicism.ridicule": "相手の能力や言動を低く扱い、笑いものにしています。",
    "cynicism.implicit_target": "相手の理解力や反応を見下し、恥をかかせる言い方です。",
    "cynicism.mocking_reaction": "相手の言動に対する嫌悪や嘲笑を、誇張した反応で示しています。",
    "cynicism.patronizing": "相手を子ども扱いするような褒め方で見下しています。",
    "cynicism.dismissive": "相手の意見や努力を取り合わず、突き放しています。",
}

_COMMUNITY_CYNICISM_RULES = frozenset(
    {
        "cynicism.uo_reaction",
        "cynicism.dowa_reaction",
        "cynicism.meu_ending",
        "cynicism.kuiya_reaction",
    }
)


class RuleConfigurationError(ValueError):
    """Raised when the moderation rule file is invalid."""


@dataclass(frozen=True)
class _CompiledRule:
    rule_id: str
    category: str
    score: int
    pattern: Pattern[str]


def normalize_text(text: str) -> str:
    """Normalize common visual obfuscations without retaining the input."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _CONTROL_CHARACTERS.sub("", normalized)
    normalized = _VARIATION_SELECTORS.sub("", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


class ModerationEngine:
    """Score text with local JSON rules and return content-free results."""

    def __init__(
        self,
        threshold: int,
        categories: Mapping[str, str],
        rules: Sequence[_CompiledRule],
        exceptions: Sequence[Pattern[str]],
    ) -> None:
        if threshold <= 0:
            raise RuleConfigurationError("threshold must be a positive integer")
        self._threshold = threshold
        self._categories = dict(categories)
        self._rules = tuple(rules)
        self._exceptions = tuple(exceptions)

    @classmethod
    def from_json(cls, path: Path) -> "ModerationEngine":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuleConfigurationError(f"failed to read rule configuration: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuleConfigurationError("rule configuration root must be an object")

        threshold = payload.get("threshold")
        if not isinstance(threshold, int) or isinstance(threshold, bool):
            raise RuleConfigurationError("threshold must be an integer")

        raw_categories = payload.get("categories")
        if not isinstance(raw_categories, dict) or not raw_categories:
            raise RuleConfigurationError("categories must be a non-empty object")

        categories: Dict[str, str] = {}
        for category, data in raw_categories.items():
            if not isinstance(category, str) or not isinstance(data, dict):
                raise RuleConfigurationError("each category must be an object")
            label = data.get("label")
            if not isinstance(label, str) or not label.strip():
                raise RuleConfigurationError(f"category {category!r} requires a label")
            categories[category] = label.strip()

        exceptions = cls._compile_exceptions(payload.get("exceptions", []))
        rules = cls._compile_rules(payload.get("rules"), categories)
        return cls(threshold, categories, rules, exceptions)

    @staticmethod
    def _compile_exceptions(raw_exceptions: object) -> List[Pattern[str]]:
        if not isinstance(raw_exceptions, list):
            raise RuleConfigurationError("exceptions must be an array")

        compiled: List[Pattern[str]] = []
        for index, pattern in enumerate(raw_exceptions):
            if not isinstance(pattern, str) or not pattern:
                raise RuleConfigurationError(f"exception {index} must be a non-empty string")
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                raise RuleConfigurationError(f"invalid exception regex at index {index}: {exc}") from exc
        return compiled

    @staticmethod
    def _compile_rules(
        raw_rules: object, categories: Mapping[str, str]
    ) -> List[_CompiledRule]:
        if not isinstance(raw_rules, list) or not raw_rules:
            raise RuleConfigurationError("rules must be a non-empty array")

        compiled: List[_CompiledRule] = []
        seen_ids = set()
        for index, data in enumerate(raw_rules):
            if not isinstance(data, dict):
                raise RuleConfigurationError(f"rule {index} must be an object")

            rule_id = data.get("id")
            category = data.get("category")
            score = data.get("score")
            pattern = data.get("pattern")

            if not isinstance(rule_id, str) or not rule_id:
                raise RuleConfigurationError(f"rule {index} requires a non-empty id")
            if rule_id in seen_ids:
                raise RuleConfigurationError(f"duplicate rule id: {rule_id}")
            if category not in categories:
                raise RuleConfigurationError(f"rule {rule_id!r} has an unknown category")
            if not isinstance(score, int) or isinstance(score, bool) or score <= 0:
                raise RuleConfigurationError(f"rule {rule_id!r} requires a positive score")
            if not isinstance(pattern, str) or not pattern:
                raise RuleConfigurationError(f"rule {rule_id!r} requires a regex pattern")

            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise RuleConfigurationError(f"invalid regex for rule {rule_id!r}: {exc}") from exc

            seen_ids.add(rule_id)
            compiled.append(_CompiledRule(rule_id, category, score, regex))
        return compiled

    def analyze(self, text: str) -> DetectionResult:
        normalized = normalize_text(text)
        if not normalized:
            return DetectionResult.empty()

        candidates = self._candidate_forms(normalized)
        if self._matches_any(self._exceptions, candidates):
            return DetectionResult.empty()

        scores, matched_ids = self._score_candidates(candidates)
        detections: List[CategoryDetection] = []
        for category, label in self._categories.items():
            score = min(scores[category], 100)
            if score >= self._threshold:
                detections.append(
                    CategoryDetection(
                        category=category,
                        label=label,
                        score=score,
                        rule_ids=tuple(matched_ids[category]),
                        reason=_local_reason(category, matched_ids[category]),
                    )
                )

        return DetectionResult(detected=bool(detections), detections=tuple(detections))

    def has_moderation_signal(self, text: str) -> bool:
        """Return whether a partial local rule warrants expensive AI review."""
        normalized = normalize_text(text)
        if not normalized:
            return False
        candidates = self._candidate_forms(normalized)
        if self._matches_any(self._exceptions, candidates):
            return False
        _, matched_ids = self._score_candidates(candidates)
        return any(matched_ids.values())

    def _score_candidates(
        self, candidates: Sequence[str]
    ) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        scores = {category: 0 for category in self._categories}
        matched_ids: Dict[str, List[str]] = {category: [] for category in self._categories}

        for rule in self._rules:
            if any(rule.pattern.search(candidate) for candidate in candidates):
                scores[rule.category] += rule.score
                matched_ids[rule.category].append(rule.rule_id)

        return scores, matched_ids

    @staticmethod
    def _candidate_forms(normalized: str) -> Tuple[str, ...]:
        compact = normalized.replace(" ", "")
        deobfuscated = _OBFUSCATION_SEPARATORS.sub("", compact)
        return tuple(dict.fromkeys((normalized, compact, deobfuscated)))

    @staticmethod
    def _matches_any(patterns: Iterable[Pattern[str]], candidates: Sequence[str]) -> bool:
        return any(pattern.search(candidate) for pattern in patterns for candidate in candidates)


def _local_reason(category: str, rule_ids: Sequence[str]) -> str:
    for rule_id in rule_ids:
        detailed = _LOCAL_RULE_REASONS.get(rule_id)
        if detailed:
            return detailed
    if category == "cynicism" and _COMMUNITY_CYNICISM_RULES.intersection(rule_ids):
        return "サーバーで冷笑として指定された反応・語尾が含まれています。"
    return _LOCAL_REASONS[category]
