import unittest

from discord_moderation_bot.config import BotConfig, ConfigError


class BotConfigTests(unittest.TestCase):
    def test_parses_multiple_guilds_and_per_guild_channels(self) -> None:
        config = BotConfig.from_env(
            {
                "DISCORD_BOT_TOKEN": "secret-value",
                "DISCORD_GUILD_IDS": "100, 101,100",
                "MONITORED_CHANNEL_IDS": "200, 201,200",
                "ALERT_CHANNEL_IDS": "100:300,101:301",
            }
        )
        self.assertEqual(config.token, "secret-value")
        self.assertEqual(config.guild_ids, frozenset({100, 101}))
        self.assertEqual(config.monitored_channel_ids, frozenset({200, 201}))
        self.assertEqual(config.alert_channel_for(100), 300)
        self.assertEqual(config.alert_channel_for(101), 301)

    def test_legacy_single_guild_values_remain_supported(self) -> None:
        config = BotConfig.from_env(
            {
                "DISCORD_BOT_TOKEN": "secret-value",
                "DISCORD_GUILD_ID": "100",
                "ALERT_CHANNEL_ID": "300",
            }
        )
        self.assertEqual(config.guild_ids, frozenset({100}))
        self.assertEqual(config.alert_channel_for(100), 300)
        self.assertFalse(config.voice_enabled)

    def test_empty_optional_values_use_defaults(self) -> None:
        config = BotConfig.from_env(
            {
                "DISCORD_BOT_TOKEN": "secret-value",
                "DISCORD_GUILD_IDS": "100",
            }
        )
        self.assertEqual(config.monitored_channel_ids, frozenset())
        self.assertIsNone(config.alert_channel_for(100))
        self.assertEqual(config.voice_channel_ids, {})
        self.assertEqual(config.voice_min_rms, 80)
        self.assertEqual(config.voice_min_utterance_ms, 180)
        self.assertEqual(config.groq_cynicism_confidence_threshold, 80)

    def test_voice_configuration_is_parsed(self) -> None:
        config = BotConfig.from_env(
            {
                "DISCORD_BOT_TOKEN": "secret-value",
                "DISCORD_GUILD_IDS": "100,101",
                "ALERT_CHANNEL_IDS": "101:301",
                "VOICE_ENABLED": "true",
                "MODERATION_BACKEND": "groq",
                "GROQ_API_KEY": "groq-secret",
                "VOICE_CHANNEL_IDS": "100:400,101:401",
                "VOICE_ALERT_CHANNEL_IDS": "100:300",
            }
        )
        self.assertTrue(config.voice_enabled)
        self.assertEqual(config.voice_channel_ids, {100: 400, 101: 401})
        self.assertEqual(config.voice_alert_channel_for(100), 300)
        self.assertEqual(config.voice_alert_channel_for(101), 301)
        self.assertEqual(config.moderation_backend, "groq")
        self.assertEqual(config.groq_speech_model, "whisper-large-v3")

    def test_all_guilds_and_automatic_voice_need_no_id_mappings(self) -> None:
        config = BotConfig.from_env(
            {
                "DISCORD_BOT_TOKEN": "secret-value",
                "MONITOR_ALL_GUILDS": "true",
                "VOICE_ENABLED": "true",
                "VOICE_AUTO_JOIN": "true",
                "MODERATION_BACKEND": "groq",
                "GROQ_API_KEY": "groq-secret",
            }
        )
        self.assertTrue(config.monitor_all_guilds)
        self.assertEqual(config.guild_ids, frozenset())
        self.assertTrue(config.voice_auto_join)
        self.assertTrue(config.is_guild_monitored(999))

    def test_automatic_voice_requires_voice_to_be_enabled(self) -> None:
        with self.assertRaisesRegex(ConfigError, "requires"):
            BotConfig.from_env(
                {
                    "DISCORD_BOT_TOKEN": "secret-value",
                    "DISCORD_GUILD_IDS": "100",
                    "VOICE_AUTO_JOIN": "true",
                }
            )

    def test_voice_requires_groq_channels_and_alert_destination(self) -> None:
        common = {
            "DISCORD_BOT_TOKEN": "secret-value",
            "DISCORD_GUILD_IDS": "100",
            "VOICE_ENABLED": "true",
        }
        for additions in (
            {},
            {"MODERATION_BACKEND": "groq"},
            {
                "MODERATION_BACKEND": "groq",
                "GROQ_API_KEY": "groq-secret",
                "VOICE_CHANNEL_IDS": "100:400",
            },
        ):
            with self.subTest(additions=additions):
                with self.assertRaises(ConfigError):
                    BotConfig.from_env({**common, **additions})

    def test_groq_models_threshold_and_timeout_are_configurable(self) -> None:
        config = BotConfig.from_env(
            {
                "DISCORD_BOT_TOKEN": "secret-value",
                "DISCORD_GUILD_IDS": "100",
                "MODERATION_BACKEND": "groq",
                "GROQ_API_KEY": "groq-secret",
                "GROQ_TEXT_MODEL": "text-model",
                "GROQ_SPEECH_MODEL": "speech-model",
                "GROQ_CONFIDENCE_THRESHOLD": "42",
                "GROQ_CYNICISM_CONFIDENCE_THRESHOLD": "81",
                "GROQ_TIMEOUT_SECONDS": "12.5",
                "VOICE_CHUNK_SECONDS": "15",
                "CONTEXT_MESSAGE_COUNT": "4",
                "CONTEXT_TTL_SECONDS": "240",
                "VOICE_MIN_RMS": "220",
                "VOICE_MIN_UTTERANCE_MS": "500",
            }
        )
        self.assertEqual(config.groq_text_model, "text-model")
        self.assertEqual(config.groq_speech_model, "speech-model")
        self.assertEqual(config.groq_confidence_threshold, 42)
        self.assertEqual(config.groq_cynicism_confidence_threshold, 81)
        self.assertEqual(config.groq_timeout_seconds, 12.5)
        self.assertEqual(config.voice_chunk_seconds, 15)
        self.assertEqual(config.context_message_count, 4)
        self.assertEqual(config.context_ttl_seconds, 240)
        self.assertEqual(config.voice_min_rms, 220)
        self.assertEqual(config.voice_min_utterance_ms, 500)

    def test_channel_mapping_must_reference_a_target_guild(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not listed"):
            BotConfig.from_env(
                {
                    "DISCORD_BOT_TOKEN": "secret-value",
                    "DISCORD_GUILD_IDS": "100",
                    "ALERT_CHANNEL_IDS": "999:300",
                }
            )

    def test_legacy_alert_is_rejected_for_multiple_guilds(self) -> None:
        with self.assertRaisesRegex(ConfigError, "one guild"):
            BotConfig.from_env(
                {
                    "DISCORD_BOT_TOKEN": "secret-value",
                    "DISCORD_GUILD_IDS": "100,101",
                    "ALERT_CHANNEL_ID": "300",
                }
            )

    def test_missing_token_does_not_echo_other_values(self) -> None:
        with self.assertRaisesRegex(ConfigError, "DISCORD_BOT_TOKEN is required"):
            BotConfig.from_env({"DISCORD_GUILD_IDS": "100"})

    def test_invalid_id_and_boolean_are_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            BotConfig.from_env(
                {"DISCORD_BOT_TOKEN": "secret-value", "DISCORD_GUILD_IDS": "bad"}
            )
        with self.assertRaisesRegex(ConfigError, "true or false"):
            BotConfig.from_env(
                {
                    "DISCORD_BOT_TOKEN": "secret-value",
                    "DISCORD_GUILD_IDS": "100",
                    "VOICE_ENABLED": "perhaps",
                }
            )


if __name__ == "__main__":
    unittest.main()
