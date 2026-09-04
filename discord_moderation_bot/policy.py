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
    source_excerpt: str = "",
) -> str:
    if not detections:
        raise ValueError("at least one detection is required")

    category_text = "・".join(item.label for item in detections)
    text = (
        "⚠️ モデレーション対象表現の可能性を検知しました"
        f"（種別：{category_text}）\n"
        f"発言者：{_safe_inline(author_name, '不明なユーザー', 80)}\n"
        f"該当ワード／発言：{_safe_excerpt(source_excerpt)}\n"
        f"問題だった点：\n{_problem_summary(detections)}\n"
        "互いを尊重した表現に言い換えてください。"
    )
    if jump_url:
        text += f"\n元メッセージ: {jump_url}"
    return text


def build_voice_alert_text(
    detections: Sequence[CategoryDetection],
    speaker_name: str,
    source_excerpt: str = "",
) -> str:
    if not detections:
        raise ValueError("at least one detection is required")
    category_text = "・".join(item.label for item in detections)
    return (
        "⚠️ VCでモデレーション対象表現の可能性を検知しました"
        f"（種別：{category_text}）\n"
        f"発言者：{_safe_inline(speaker_name, '不明なユーザー', 80)}\n"
        f"認識したワード／発言：{_safe_excerpt(source_excerpt)}\n"
        f"問題だった点：\n{_problem_summary(detections)}\n"
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
        reason = _safe_inline(_specific_reason(item) or item.reason, fallback, 200)
        parts.append(f"・{item.label}：{reason}")
    return "\n".join(parts)


def _specific_reason(item: CategoryDetection) -> str:
    rule_ids = set(item.rule_ids)
    if "cynicism.uo_reaction" in rule_ids:
        return "「うお」単体またはその笑い表記は、このサーバーで相手を茶化す冷笑反応として指定されています。"
    if "cynicism.dowa_reaction" in rule_ids:
        return "「どわー」系の反応は、このサーバーで相手を茶化す冷笑反応として指定されています。"
    if "cynicism.meu_ending" in rule_ids:
        return "語尾の「めう」は、このサーバーで相手を茶化す冷笑表現として指定されています。"
    if "cynicism.kuiya_reaction" in rule_ids:
        return "「クイヤ」は、このサーバーで冷笑表現として指定されている語です。"
    if rule_ids.intersection(
        {"discrimination.gendered_emasculation", "cynicism.gendered_mockery"}
    ):
        return "「女々しい」など、性別役割を押しつけて相手を弱い・劣る存在として扱う表現です。"
    if "sexual_content.explicit_or_vulgar" in rule_ids:
        return "露骨な性的語、性行為・身体部位への言及、または下ネタとして扱われる表現です。"
    if "drug_content.illicit_or_abuse" in rule_ids:
        return "違法・娯楽目的の薬物名、乱用、売買、勧誘または摂取に関係する表現です。"
    if "sensitive_term.adhd" in rule_ids:
        return "「ADHD」は、差別と断定せず確認対象にするサーバー指定の要注意語です。"
    if "discrimination.explicit_slur" in rule_ids:
        return "人種・民族・障害・性的指向などの属性を傷つける蔑称として使われる語を含みます。"
    if "discrimination.gender_or_orientation_slur" in rule_ids:
        return "性別・性自認・性的指向を理由に相手を侮辱または排除する表現です。"
    if "cynicism.direct_abuse" in rule_ids:
        return "相手へ危害や消失を望む言葉、または強く黙らせる直接的な暴言です。"
    if "cynicism.english_targeted_insult" in rule_ids:
        return "英語で相手の知性・能力・人格を直接見下す表現です。"
    if "drug_content.coded_action" in rule_ids:
        return "一般語を薬物の隠語として用い、摂取・売買・入手に結びつける表現です。"
    if any(rule_id.startswith("discrimination.") for rule_id in rule_ids):
        return "属性や立場を理由に、集団を一般化・劣等視・排除する意味を含む表現です。"
    if any(rule_id.startswith("cynicism.") for rule_id in rule_ids):
        return "相手を見下す、失敗を笑う、または突き放す冷笑として受け取られる表現です。"
    return ""


def _safe_inline(value: str, fallback: str, limit: int) -> str:
    cleaned = " ".join(str(value).split()).strip()
    cleaned = cleaned.replace("@", "＠").replace("`", "'")
    return (cleaned or fallback)[:limit]


def _safe_excerpt(value: str, limit: int = 240) -> str:
    cleaned = " ".join(str(value).split()).strip()
    for unsafe, replacement in (
        ("@", "＠"),
        ("`", "'"),
        ("*", "＊"),
        ("_", "＿"),
        ("~", "〜"),
        ("|", "｜"),
    ):
        cleaned = cleaned.replace(unsafe, replacement)
    if not cleaned:
        return "（取得できませんでした）"
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return f"「{cleaned}」"
