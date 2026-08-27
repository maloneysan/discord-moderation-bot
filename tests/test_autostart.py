from pathlib import Path
import plistlib
import unittest


ROOT = Path(__file__).resolve().parent.parent
PLIST_PATH = ROOT / "launchd" / "io.github.maloneysan.discord-moderation-bot.plist"
RUNNER_PATH = ROOT / "scripts" / "run_from_keychain.zsh"
AUTOSTART_CONFIG_PATH = ROOT / "config" / "autostart.env.example"
INSTALLER_PATH = ROOT / "install_autostart.command"


class AutostartConfigurationTests(unittest.TestCase):
    def test_launch_agent_runs_keychain_runner_and_restarts(self) -> None:
        with PLIST_PATH.open("rb") as handle:
            payload = plistlib.load(handle)
        self.assertEqual(payload["Label"], "io.github.maloneysan.discord-moderation-bot")
        self.assertTrue(payload["RunAtLoad"])
        self.assertTrue(payload["KeepAlive"])
        self.assertGreaterEqual(payload["ThrottleInterval"], 30)
        self.assertTrue(payload["ProgramArguments"][0].endswith("run_from_keychain.zsh"))
        self.assertEqual(payload["WorkingDirectory"], "__RUNTIME_DIR__")
        self.assertNotIn("/Users/", PLIST_PATH.read_text(encoding="utf-8"))

    def test_token_is_read_from_keychain_not_configuration(self) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        config = AUTOSTART_CONFIG_PATH.read_text(encoding="utf-8")
        plist = PLIST_PATH.read_text(encoding="utf-8")
        self.assertIn("security find-generic-password", runner)
        self.assertIn("DiscordModerationBotGroqApiKey", runner)
        self.assertNotIn("DISCORD_BOT_TOKEN=", config)
        self.assertNotIn("GROQ_API_KEY=", config)
        self.assertNotIn("DISCORD_BOT_TOKEN", plist)

    def test_autostart_config_has_required_guild(self) -> None:
        values = {}
        for line in AUTOSTART_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        self.assertTrue(values["DISCORD_GUILD_IDS"])
        self.assertEqual(values["MONITOR_ALL_GUILDS"], "true")
        self.assertIn(values["VOICE_ENABLED"], {"true", "false"})
        self.assertEqual(values["VOICE_ENABLED"], "true")
        self.assertEqual(values["VOICE_AUTO_JOIN"], "true")
        self.assertEqual(values["MODERATION_BACKEND"], "groq")
        self.assertEqual(values["GROQ_TEXT_MODEL"], "openai/gpt-oss-120b")
        self.assertEqual(values["GROQ_SPEECH_MODEL"], "whisper-large-v3")

    def test_installer_retries_launch_agent_registration(self) -> None:
        installer = INSTALLER_PATH.read_text(encoding="utf-8")
        self.assertIn("if ! /bin/launchctl bootstrap", installer)
        self.assertIn("/bin/sleep 2", installer)


if __name__ == "__main__":
    unittest.main()
