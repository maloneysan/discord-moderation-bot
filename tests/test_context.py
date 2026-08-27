import unittest

from discord_moderation_bot.context import ConversationContextBuffer


class ConversationContextBufferTests(unittest.TestCase):
    def test_keeps_only_recent_messages_without_identifiers(self) -> None:
        buffer = ConversationContextBuffer(max_messages=3, ttl_seconds=180)
        for message_id, text in enumerate(("一件目", "二件目", "三件目", "四件目"), 1):
            buffer.remember(100, message_id, text, now=float(message_id))

        self.assertEqual(
            buffer.recent(100, now=5),
            ("二件目", "三件目", "四件目"),
        )
        self.assertNotIn("author", repr(buffer.__dict__))

    def test_edit_replaces_message_and_current_message_can_be_excluded(self) -> None:
        buffer = ConversationContextBuffer(max_messages=3, ttl_seconds=180)
        buffer.remember(100, 1, "編集前", now=1)
        buffer.remember(100, 1, "編集後", now=2)
        buffer.remember(100, 2, "現在", now=3)

        self.assertEqual(
            buffer.recent(100, excluding_message_id=2, now=3),
            ("編集後",),
        )

    def test_expired_content_is_removed_and_clear_drops_everything(self) -> None:
        buffer = ConversationContextBuffer(max_messages=3, ttl_seconds=10)
        buffer.remember(100, 1, "期限切れ", now=1)
        self.assertEqual(buffer.recent(100, now=12), ())
        buffer.remember(100, 2, "一時情報", now=13)
        buffer.clear()
        self.assertEqual(buffer.message_count, 0)


if __name__ == "__main__":
    unittest.main()
