import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import AsyncMock, Mock, patch

from discord_moderation_bot.config import BotConfig
from discord_moderation_bot.engine import ModerationEngine
from discord_moderation_bot.service import LocalModerationService
from discord_moderation_bot.voice import (
    VoiceAlertRegistry,
    VoiceModerationManager,
    VoicePcmChunker,
    VoiceTranscriptModerator,
    _ensure_opus_loaded,
)


RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.json"


class OpusRuntimeTests(unittest.TestCase):
    @patch("discord_moderation_bot.voice.ctypes.util.find_library", return_value=None)
    @patch("discord_moderation_bot.voice.discord.opus.load_opus")
    @patch("discord_moderation_bot.voice.discord.opus.is_loaded")
    def test_falls_back_to_homebrew_opus(
        self, is_loaded: Mock, load_opus: Mock, _find_library: Mock
    ) -> None:
        is_loaded.side_effect = [False, True]

        _ensure_opus_loaded()

        load_opus.assert_called_once_with("/opt/homebrew/lib/libopus.dylib")

    @patch("discord_moderation_bot.voice.discord.opus.load_opus")
    @patch("discord_moderation_bot.voice.discord.opus.is_loaded", return_value=True)
    def test_does_not_reload_an_available_decoder(
        self, _is_loaded: Mock, load_opus: Mock
    ) -> None:
        _ensure_opus_loaded()
        load_opus.assert_not_called()


class VoiceAlertRegistryTests(unittest.TestCase):
    def test_per_speaker_category_cooldown(self) -> None:
        registry = VoiceAlertRegistry(cooldown_seconds=20)
        self.assertEqual(
            registry.claim_new(100, 200, ["cynicism"], now=10),
            ("cynicism",),
        )
        self.assertEqual(registry.claim_new(100, 200, ["cynicism"], now=29), ())
        self.assertEqual(
            registry.claim_new(100, 201, ["cynicism"], now=29),
            ("cynicism",),
        )
        self.assertEqual(
            registry.claim_new(100, 200, ["cynicism"], now=30),
            ("cynicism",),
        )


class VoiceTranscriptModeratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcript_is_passed_to_alert_but_not_retained(self) -> None:
        sender = AsyncMock()
        service = LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        moderator = VoiceTranscriptModerator(
            service,
            sender,
            VoiceAlertRegistry(cooldown_seconds=20),
        )
        transcript = "外国人は出ていけ"

        await moderator.process(100, 200, transcript)
        await moderator.process(100, 200, transcript)

        sender.assert_awaited_once()
        guild_id, user_id, detections, source_excerpt = sender.await_args.args
        self.assertEqual((guild_id, user_id), (100, 200))
        self.assertEqual([item.label for item in detections], ["差別表現"])
        self.assertIn("排除", detections[0].reason)
        self.assertEqual(source_excerpt, transcript)
        self.assertNotIn(transcript, repr(moderator.__dict__))

    async def test_failed_alert_is_released_for_retry(self) -> None:
        sender = AsyncMock(side_effect=[RuntimeError("send failed"), None])
        service = LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        moderator = VoiceTranscriptModerator(
            service,
            sender,
            VoiceAlertRegistry(cooldown_seconds=20),
        )
        with self.assertRaises(RuntimeError):
            await moderator.process(100, 200, "どわー")
        await moderator.process(100, 200, "どわー")
        self.assertEqual(sender.await_count, 2)


class VoicePcmChunkerTests(unittest.TestCase):
    @staticmethod
    def voiced_pcm(frame_count: int) -> bytes:
        return b"\xe8\x03\xe8\x03" * frame_count

    def test_converts_discord_pcm_to_memory_only_wav_chunk(self) -> None:
        chunker = VoicePcmChunker(chunk_seconds=1)
        discord_pcm = self.voiced_pcm(48_000)

        chunks = chunker.accept_pcm(100, 200, discord_pcm)

        self.assertEqual(len(chunks), 1)
        with wave.open(io.BytesIO(chunks[0]), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16_000)
            self.assertEqual(wav_file.getnframes(), 16_000)

    def test_forget_discards_partial_speaker_audio(self) -> None:
        chunker = VoicePcmChunker(chunk_seconds=1)
        half_second = self.voiced_pcm(24_000)
        self.assertEqual(chunker.accept_pcm(100, 200, half_second), ())
        chunker.forget(100, 200)
        self.assertEqual(chunker.accept_pcm(100, 200, half_second), ())

    def test_flush_emits_a_short_utterance_without_waiting_for_full_chunk(self) -> None:
        chunker = VoicePcmChunker(chunk_seconds=10)
        half_second = self.voiced_pcm(24_000)
        self.assertEqual(chunker.accept_pcm(100, 200, half_second), ())

        wav_audio = chunker.flush(100, 200)

        self.assertIsNotNone(wav_audio)
        with wave.open(io.BytesIO(wav_audio), "rb") as wav_file:
            self.assertGreater(wav_file.getnframes(), 0)
        self.assertIsNone(chunker.flush(100, 200))

    def test_silence_flushes_buffer_even_without_speaking_stop_event(self) -> None:
        chunker = VoicePcmChunker(chunk_seconds=10)
        half_second = self.voiced_pcm(24_000)
        silence_frame = b"\x00\x00" * 2 * 960

        self.assertEqual(chunker.accept_pcm(100, 200, half_second), ())
        chunks = chunker.accept_pcm(100, 200, silence_frame)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunker.flush(100, 200))

    def test_lower_default_rms_keeps_clear_low_volume_voice(self) -> None:
        sample = int(60).to_bytes(2, "little", signed=True)
        low_volume_voice = sample * 2 * 24_000
        silence_frame = b"\x00\x00" * 2 * 960

        permissive = VoicePcmChunker(chunk_seconds=10)
        strict = VoicePcmChunker(chunk_seconds=10, min_rms=80)
        self.assertEqual(permissive.accept_pcm(100, 200, low_volume_voice), ())
        self.assertEqual(len(permissive.accept_pcm(100, 200, silence_frame)), 1)
        self.assertEqual(strict.accept_pcm(100, 200, low_volume_voice), ())
        self.assertIsNone(strict.flush(100, 200))

    def test_silence_and_extremely_short_audio_are_discarded(self) -> None:
        chunker = VoicePcmChunker(chunk_seconds=10, min_utterance_ms=350)
        silence = b"\x00\x00" * 2 * 48_000
        self.assertEqual(chunker.accept_pcm(100, 200, silence), ())
        self.assertIsNone(chunker.flush(100, 200))

        short_voice = self.voiced_pcm(4_800)
        self.assertEqual(chunker.accept_pcm(100, 200, short_voice), ())
        self.assertIsNone(chunker.flush(100, 200))
        self.assertGreaterEqual(chunker.discarded_for_guild(100), 2)


class AutomaticVoiceMonitoringTests(unittest.IsolatedAsyncioTestCase):
    def make_manager(self, client) -> VoiceModerationManager:
        config = BotConfig(
            token="not-a-real-token",
            guild_ids=frozenset(),
            monitored_channel_ids=frozenset(),
            alert_channel_ids={},
            rules_path=RULES_PATH,
            monitor_all_guilds=True,
            voice_enabled=True,
            voice_auto_join=True,
            moderation_backend="groq",
            groq_api_key="groq-secret",
        )
        service = LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        return VoiceModerationManager(
            client,
            config,
            service,
            AsyncMock(),
        )

    @staticmethod
    def channel(channel_id: int, human_count: int):
        humans = [SimpleNamespace(bot=False) for _ in range(human_count)]
        return SimpleNamespace(id=channel_id, members=humans)

    async def test_selects_the_busiest_active_vc(self) -> None:
        client = SimpleNamespace(user=SimpleNamespace(id=999))
        manager = self.make_manager(client)
        manager._connect_guild = AsyncMock()
        quiet = self.channel(400, 1)
        busy = self.channel(401, 3)
        guild = SimpleNamespace(
            id=100,
            voice_client=None,
            voice_channels=[quiet, busy],
            stage_channels=[],
        )

        await manager._sync_auto_guild(guild)

        manager._connect_guild.assert_awaited_once_with(100, 401)

    async def test_keeps_current_vc_while_humans_remain(self) -> None:
        client = SimpleNamespace(user=SimpleNamespace(id=999))
        manager = self.make_manager(client)
        manager._connect_guild = AsyncMock()
        current = self.channel(400, 1)
        voice_client = SimpleNamespace(channel=current)
        guild = SimpleNamespace(
            id=100,
            voice_client=voice_client,
            voice_channels=[current, self.channel(401, 5)],
            stage_channels=[],
        )

        await manager._sync_auto_guild(guild)

        manager._connect_guild.assert_not_awaited()
        self.assertIs(manager._sessions[100], voice_client)

    async def test_disconnects_when_no_humans_remain(self) -> None:
        client = SimpleNamespace(user=SimpleNamespace(id=999))
        manager = self.make_manager(client)
        empty = self.channel(400, 0)
        voice_client = SimpleNamespace(
            channel=empty,
            stop_listening=Mock(),
            disconnect=AsyncMock(),
        )
        guild = SimpleNamespace(
            id=100,
            voice_client=voice_client,
            voice_channels=[empty],
            stage_channels=[],
        )

        await manager._sync_auto_guild(guild)

        voice_client.stop_listening.assert_called_once()
        voice_client.disconnect.assert_awaited_once_with(force=True)

    async def test_human_join_event_triggers_automatic_sync(self) -> None:
        client = SimpleNamespace(user=SimpleNamespace(id=999))
        manager = self.make_manager(client)
        manager._sync_auto_guild = AsyncMock()
        active = self.channel(400, 1)
        guild = SimpleNamespace(id=100)
        member = SimpleNamespace(id=200, bot=False, guild=guild)

        await manager.handle_voice_state_update(
            member,
            SimpleNamespace(channel=None),
            SimpleNamespace(channel=active),
        )

        manager._sync_auto_guild.assert_awaited_once_with(guild, active)

    async def test_disabling_auto_join_disconnects_current_vc(self) -> None:
        voice_client = SimpleNamespace(
            channel=self.channel(400, 1),
            stop_listening=Mock(),
            disconnect=AsyncMock(),
            is_connected=Mock(return_value=True),
        )
        guild = SimpleNamespace(id=100, voice_client=voice_client)
        client = SimpleNamespace(
            user=SimpleNamespace(id=999),
            get_guild=Mock(return_value=guild),
        )
        manager = self.make_manager(client)

        await manager.set_auto_join_for_guild(100, False)

        self.assertFalse(manager.is_auto_join_enabled(100))
        voice_client.stop_listening.assert_called_once()
        voice_client.disconnect.assert_awaited_once_with(force=True)

    async def test_stale_voice_client_is_reconnected(self) -> None:
        stale = SimpleNamespace(
            channel=self.channel(400, 1),
            stop_listening=Mock(),
            disconnect=AsyncMock(),
            is_connected=Mock(return_value=False),
        )
        target = self.channel(401, 2)
        guild = SimpleNamespace(
            id=100,
            voice_client=stale,
            voice_channels=[target],
            stage_channels=[],
        )
        client = SimpleNamespace(user=SimpleNamespace(id=999))
        manager = self.make_manager(client)
        manager._connect_guild = AsyncMock()

        await manager._sync_auto_guild(guild)

        stale.disconnect.assert_awaited_once_with(force=True)
        manager._connect_guild.assert_awaited_once_with(100, 401)

    async def test_start_resynchronizes_after_gateway_reconnect(self) -> None:
        guild = SimpleNamespace(id=100)
        client = SimpleNamespace(user=SimpleNamespace(id=999), guilds=[guild])
        manager = self.make_manager(client)
        manager._started = True
        manager.sync_guild = AsyncMock()

        await manager.start()

        manager.sync_guild.assert_awaited_once_with(guild)

    async def test_monitoring_notice_is_sent_once_per_vc_session(self) -> None:
        client = SimpleNamespace(user=SimpleNamespace(id=999))
        notice_sender = AsyncMock()
        config = self.make_manager(client)._config
        service = LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        manager = VoiceModerationManager(
            client, config, service, AsyncMock(), notice_sender
        )

        await manager._announce_monitoring(100, 400)
        await manager._announce_monitoring(100, 400)

        notice_sender.assert_awaited_once()
        self.assertIn("保存しません", notice_sender.await_args.args[2])

    async def test_health_notice_is_rate_limited_and_status_has_no_content(self) -> None:
        client = SimpleNamespace(user=SimpleNamespace(id=999))
        notice_sender = AsyncMock()
        config = self.make_manager(client)._config
        service = LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        manager = VoiceModerationManager(
            client, config, service, AsyncMock(), notice_sender
        )

        await manager._record_error(100, "音声認識に失敗しました")
        await manager._record_error(100, "音声認識に失敗しました")

        notice_sender.assert_awaited_once()
        status = manager.status_summary(100)
        self.assertIn("直近エラー", status)
        self.assertNotIn("発言内容", status)


if __name__ == "__main__":
    unittest.main()
