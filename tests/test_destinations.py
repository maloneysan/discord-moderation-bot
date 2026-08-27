from pathlib import Path
import stat
import tempfile
import unittest

from discord_moderation_bot.destinations import AlertChannelStore


class AlertChannelStoreTests(unittest.TestCase):
    def test_persists_only_guild_and_channel_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "destinations.json"
            store = AlertChannelStore(path)

            store.set(100, 300)
            reloaded = AlertChannelStore(path)

            self.assertEqual(reloaded.snapshot(), {100: 300})
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("message", content.casefold())
            self.assertNotIn("transcript", content.casefold())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_clear_is_persistent_and_reports_whether_value_existed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "destinations.json"
            store = AlertChannelStore(path)
            store.set(100, 300)

            self.assertTrue(store.clear(100))
            self.assertFalse(store.clear(100))
            self.assertEqual(AlertChannelStore(path).snapshot(), {})

    def test_invalid_state_uses_empty_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "destinations.json"
            path.write_text("not-json", encoding="utf-8")

            store = AlertChannelStore(path)

            self.assertEqual(store.snapshot(), {})


if __name__ == "__main__":
    unittest.main()
