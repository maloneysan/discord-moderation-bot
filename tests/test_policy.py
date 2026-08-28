import unittest

from discord_moderation_bot.models import CategoryDetection
from discord_moderation_bot.policy import (
    MessageContext,
    build_alert_text,
    build_voice_alert_text,
    should_monitor_message,
)


def detection(category: str, label: str, reason: str) -> CategoryDetection:
    return CategoryDetection(category, label, 90, ("test.rule",), reason)


def make_context(**overrides) -> MessageContext:
    values = {
        "guild_id": 100,
        "channel_id": 200,
        "parent_channel_id": None,
        "author_is_bot": False,
        "webhook_id": None,
        "has_text": True,
    }
    values.update(overrides)
    return MessageContext(**values)


class MessagePolicyTests(unittest.TestCase):
    def test_normal_messages_in_either_expected_guild_are_monitored(self) -> None:
        guilds = frozenset({100, 101})
        self.assertTrue(should_monitor_message(make_context(), guilds, frozenset()))
        self.assertTrue(
            should_monitor_message(make_context(guild_id=101), guilds, frozenset())
        )

    def test_bot_webhook_dm_other_guild_and_empty_messages_are_ignored(self) -> None:
        ignored = [
            make_context(author_is_bot=True),
            make_context(webhook_id=999),
            make_context(guild_id=None),
            make_context(guild_id=102),
            make_context(has_text=False),
        ]
        for context in ignored:
            with self.subTest(context=context):
                self.assertFalse(
                    should_monitor_message(context, frozenset({100, 101}), frozenset())
                )

    def test_channel_allowlist_and_thread_parent_are_supported(self) -> None:
        allowed = frozenset({200})
        guilds = frozenset({100})
        self.assertTrue(should_monitor_message(make_context(), guilds, allowed))
        self.assertTrue(
            should_monitor_message(
                make_context(channel_id=250, parent_channel_id=200), guilds, allowed
            )
        )
        self.assertFalse(
            should_monitor_message(
                make_context(channel_id=251, parent_channel_id=201), guilds, allowed
            )
        )

    def test_all_guild_mode_accepts_any_joined_guild_but_not_dm(self) -> None:
        self.assertTrue(
            should_monitor_message(
                make_context(guild_id=999),
                frozenset(),
                frozenset(),
                monitor_all_guilds=True,
            )
        )
        self.assertFalse(
            should_monitor_message(
                make_context(guild_id=None),
                frozenset(),
                frozenset(),
                monitor_all_guilds=True,
            )
        )

    def test_alert_does_not_include_mentions_or_source_content(self) -> None:
        alert = build_alert_text(
            [
                detection("discrimination", "差別表現", "属性を理由に排除する内容です。"),
                detection("cynicism", "冷笑", "相手を嘲笑する内容です。"),
            ],
            "＠everyone `話者`",
        )
        self.assertNotIn("@", alert)
        self.assertNotIn("外国人は出ていけ", alert)
        self.assertIn("差別表現・冷笑", alert)
        self.assertIn("発言者：＠everyone '話者'", alert)
        self.assertIn("問題だった点：", alert)
        self.assertIn("・差別表現：", alert)
        self.assertIn("属性を理由に排除", alert)

    def test_dedicated_alert_includes_only_supplied_jump_url(self) -> None:
        url = "https://discord.com/channels/100/200/300"
        alert = build_alert_text(
            [detection("cynicism", "冷笑", "相手を嘲笑する内容です。")],
            "話者",
            jump_url=url,
        )
        self.assertIn(url, alert)

    def test_new_categories_use_neutral_moderation_heading(self) -> None:
        alert = build_alert_text(
            [
                detection("sexual_content", "性的表現", "下ネタとして扱われる内容です。"),
                detection(
                    "sensitive_term",
                    "要注意語（ADHD）",
                    "サーバー指定語への言及です。",
                ),
                detection(
                    "drug_content",
                    "薬物関連",
                    "違法薬物や薬物乱用に関連する内容です。",
                ),
            ],
            "話者",
        )
        self.assertIn("モデレーション対象表現", alert)
        self.assertIn("性的表現・要注意語（ADHD）・薬物関連", alert)
        self.assertNotIn("差別表現の可能性", alert)

    def test_voice_alert_has_no_speaker_mention_or_transcript(self) -> None:
        alert = build_voice_alert_text(
            [detection("discrimination", "差別表現", "属性への排除です。")],
            "VC話者",
        )
        self.assertIn("VC", alert)
        self.assertIn("発言者：VC話者", alert)
        self.assertIn("問題だった点：", alert)
        self.assertIn("・差別表現：", alert)
        self.assertNotIn("@", alert)
        self.assertNotIn("外国人は出ていけ", alert)


if __name__ == "__main__":
    unittest.main()
