from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence

from .models import CategoryDetection


@dataclass(frozen=True)
class MessageContext:
    guild_id: Optional[int]
    channel_id: int
    parent_channel_id: Optional[int]
    author_is_bot: bool
    webhook_id: Optional[int]
    has_text: bool


def should_monitor_message(
    context: MessageContext,
    expected_guild_ids: FrozenSet[int],
    monitored_channel_ids: FrozenSet[int],
    monitor_all_guilds: bool = False,
) -> bool:
    if context.guild_id is None:
        return False
    if not monitor_all_guilds and context.guild_id not in expected_guild_ids:
        return False
    if context.author_is_bot or context.webhook_id is not None or not context.has_text:
        return False
    if not monitored_channel_ids:
        return True
    return (
        context.channel_id in monitored_channel_ids
        or context.parent_channel_id in monitored_channel_ids
    )


def build_alert_text(
    detections: Sequence[CategoryDetection],
    author_name: str,
    jump_url: Optional[str] = None,
) -> str:
    if not detections:
        raise ValueError("at least one detection is required")

    category_text = "・".join(item.label for item in detections)
    text = (
        "⚠️ モデレーション対象表現の可能性を検知しました"
        f"（種別：{category_text}）\n"
        f"発言者：{_safe_inline(author_name, '不明なユーザー', 80)}\n"
        f"問題点：{_problem_summary(detections)}\n"
        "互いを尊重した表現に言い換えてください。"
    )
    if jump_url:
        text += f"\n元メッセージ: {jump_url}"
    return text


def build_voice_alert_text(
    detections: Sequence[CategoryDetection], speaker_name: str
) -> str:
    if not detections:
        raise ValueError("at least one detection is required")
    category_text = "・".join(item.label for item in detections)
    return (
        "⚠️ VCでモデレーション対象表現の可能性を検知しました"
        f"（種別：{category_text}）\n"
        f"発言者：{_safe_inline(speaker_name, '不明なユーザー', 80)}\n"
        f"問題点：{_problem_summary(detections)}\n"
        "互いを尊重した発言を心がけてください。"
    )


def _problem_summary(detections: Sequence[CategoryDetection]) -> str:
    parts = []
    for item in detections:
        fallback = {
            "discrimination": "属性や立場を理由に相手を不当に扱う内容です。",
            "cynicism": "相手を見下したり嘲笑する内容です。",
            "sexual_content": "性的な話題や下ネタとして扱われる内容です。",
            "sensitive_term": "サーバーで指定された要注意語への言及です。",
            "drug_content": "違法薬物や薬物乱用に関連する内容です。",
        }.get(item.category, "モデレーション対象となる内容です。")
        reason = _safe_inline(item.reason, fallback, 160)
        parts.append(f"{item.label} — {reason}")
    return " / ".join(parts)


def _safe_inline(value: str, fallback: str, limit: int) -> str:
    cleaned = " ".join(str(value).split()).strip()
    cleaned = cleaned.replace("@", "＠").replace("`", "'")
    return (cleaned or fallback)[:limit]
