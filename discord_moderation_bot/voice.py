from __future__ import annotations

import asyncio
import audioop
import ctypes.util
import importlib
import io
import logging
import threading
import time
import wave
from collections import defaultdict
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

import discord

from .config import BotConfig
from .models import CategoryDetection
from .service import ModerationService


LOGGER = logging.getLogger(__name__)
VoiceAlertSender = Callable[
    [int, int, List[CategoryDetection], str], Awaitable[None]
]
VoiceNoticeSender = Callable[[int, Optional[int], str], Awaitable[None]]


def _ensure_opus_loaded() -> None:
    """Load the native Opus decoder required for receiving PCM audio."""
    if discord.opus.is_loaded():
        return

    candidates = (
        ctypes.util.find_library("opus"),
        "/opt/homebrew/lib/libopus.dylib",
        "/usr/local/lib/libopus.dylib",
        "/opt/local/lib/libopus.dylib",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            discord.opus.load_opus(candidate)
        except OSError:
            continue
        if discord.opus.is_loaded():
            LOGGER.info("Native Opus decoder loaded")
            return
    raise RuntimeError("native Opus decoder is unavailable")


class VoiceAlertRegistry:
    """Content-free per-speaker cooldown for repeated VC alerts."""

    def __init__(self, cooldown_seconds: float = 20.0) -> None:
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        self._cooldown_seconds = cooldown_seconds
        self._last_alerts: Dict[Tuple[int, int, str], float] = {}

    def claim_new(
        self,
        guild_id: int,
        user_id: int,
        categories: Iterable[str],
        now: Optional[float] = None,
    ) -> Tuple[str, ...]:
        current = time.monotonic() if now is None else now
        claimed = []
        for category in categories:
            key = (guild_id, user_id, category)
            previous = self._last_alerts.get(key)
            if previous is not None and current - previous < self._cooldown_seconds:
                continue
            self._last_alerts[key] = current
            claimed.append(category)
        return tuple(claimed)

    def release(
        self, guild_id: int, user_id: int, categories: Iterable[str]
    ) -> None:
        for category in categories:
            self._last_alerts.pop((guild_id, user_id, category), None)


class VoiceTranscriptModerator:
    """Analyze an ephemeral transcript and pass it only to the alert sender."""

    def __init__(
        self,
        service: ModerationService,
        alert_sender: VoiceAlertSender,
        registry: Optional[VoiceAlertRegistry] = None,
    ) -> None:
        self._service = service
        self._alert_sender = alert_sender
        self._registry = registry or VoiceAlertRegistry()

    async def process(self, guild_id: int, user_id: int, transcript: str) -> int:
        result = await self._service.analyze(transcript, request_source="voice")
        if not result.detected:
            return 0

        detections = {item.category: item for item in result.detections}
        claimed = self._registry.claim_new(guild_id, user_id, detections)
        if not claimed:
            return 0
        claimed_detections = [detections[category] for category in claimed]
        try:
            await self._alert_sender(
                guild_id,
                user_id,
                claimed_detections,
                transcript,
            )
        except Exception:
            self._registry.release(guild_id, user_id, claimed)
            raise
        return len(claimed)


class VoicePcmChunker:
    """Build short in-memory 16 kHz mono WAV chunks for external recognition."""

    def __init__(
        self,
        chunk_seconds: int = 10,
        *,
        min_rms: int = 25,
        min_utterance_ms: int = 100,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        if min_rms < 0:
            raise ValueError("min_rms must not be negative")
        if min_utterance_ms <= 0:
            raise ValueError("min_utterance_ms must be positive")
        self._target_bytes = 16_000 * 2 * chunk_seconds
        self._minimum_bytes = int(16_000 * 2 * min_utterance_ms / 1000)
        self._min_rms = min_rms
        self._buffers: Dict[Tuple[int, int], bytearray] = {}
        self._rate_states: Dict[Tuple[int, int], object] = {}
        self._discarded_by_guild: Dict[int, int] = defaultdict(int)
        self._lock = threading.Lock()

    def accept_pcm(self, guild_id: int, user_id: int, pcm: bytes) -> Tuple[bytes, ...]:
        if not pcm:
            return ()
        key = (guild_id, user_id)
        with self._lock:
            mono = audioop.tomono(pcm, 2, 0.5, 0.5)
            converted, state = audioop.ratecv(
                mono,
                2,
                1,
                48_000,
                16_000,
                self._rate_states.get(key),
            )
            self._rate_states[key] = state
            if converted and audioop.rms(converted, 2) < self._min_rms:
                buffer = self._buffers.pop(key, None)
                self._rate_states.pop(key, None)
                if buffer and len(buffer) >= self._minimum_bytes:
                    return (_pcm_to_wav(bytes(buffer)),)
                self._discarded_by_guild[guild_id] += 1
                return ()
            buffer = self._buffers.setdefault(key, bytearray())
            buffer.extend(converted)
            chunks = []
            while len(buffer) >= self._target_bytes:
                pcm_chunk = bytes(buffer[: self._target_bytes])
                del buffer[: self._target_bytes]
                chunks.append(_pcm_to_wav(pcm_chunk))
            return tuple(chunks)

    def forget(self, guild_id: int, user_id: int) -> None:
        key = (guild_id, user_id)
        with self._lock:
            self._buffers.pop(key, None)
            self._rate_states.pop(key, None)

    def flush(self, guild_id: int, user_id: int) -> Optional[bytes]:
        key = (guild_id, user_id)
        with self._lock:
            buffer = self._buffers.pop(key, None)
            self._rate_states.pop(key, None)
            if not buffer:
                return None
            if len(buffer) < self._minimum_bytes:
                self._discarded_by_guild[guild_id] += 1
                return None
            return _pcm_to_wav(bytes(buffer))

    def discarded_for_guild(self, guild_id: int) -> int:
        with self._lock:
            return self._discarded_by_guild.get(guild_id, 0)

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._rate_states.clear()
            self._discarded_by_guild.clear()


def _pcm_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(pcm)
    return output.getvalue()


class VoiceModerationManager:
    """Optional Discord voice receiver backed by external speech recognition."""

    def __init__(
        self,
        client: discord.Client,
        config: BotConfig,
        service: ModerationService,
        alert_sender: VoiceAlertSender,
        notice_sender: Optional[VoiceNoticeSender] = None,
    ) -> None:
        self._client = client
        self._config = config
        self._service = service
        self._moderator = VoiceTranscriptModerator(service, alert_sender)
        self._notice_sender = notice_sender
        self._sessions: Dict[int, object] = {}
        self._chunker: Optional[VoicePcmChunker] = None
        self._voice_recv = None
        self._started = False
        self._closing = False
        self._guild_locks: Dict[int, asyncio.Lock] = {}
        self._speaker_locks: Dict[Tuple[int, int], asyncio.Lock] = {}
        self._auto_join_suppressed: Set[int] = set()
        self._runtime_available = False
        self._announced_channels: Set[Tuple[int, int]] = set()
        self._last_error_by_guild: Dict[int, str] = {}
        self._last_error_notice_at: Dict[int, float] = {}
        self._last_transcription_at: Dict[int, float] = {}
        self._processed_chunks: Dict[int, int] = defaultdict(int)
        self._rejected_chunks: Dict[int, int] = defaultdict(int)

    async def start(self) -> None:
        if not self._config.voice_enabled:
            return
        if self._started:
            await self._sync_all_auto_guilds()
            return
        try:
            self._load_runtime()
        except Exception as exc:
            LOGGER.warning("VC moderation is unavailable (error=%s)", type(exc).__name__)
            for guild in tuple(self._client.guilds):
                if self._config.is_guild_monitored(guild.id):
                    await self._record_error(guild.id, "音声受信機能を初期化できません")
            return
        self._started = True

        for guild_id, channel_id in self._config.voice_channel_ids.items():
            try:
                await self._connect_guild(guild_id, channel_id)
            except Exception as exc:
                LOGGER.warning(
                    "Could not start VC moderation (guild_id=%s, channel_id=%s, error=%s)",
                    guild_id,
                    channel_id,
                    type(exc).__name__,
                )
                await self._record_error(guild_id, "指定VCへ接続できません")

        await self._sync_all_auto_guilds()

    async def _sync_all_auto_guilds(self) -> None:
        if not self._config.voice_auto_join:
            return
        for guild in tuple(self._client.guilds):
            if not self._config.is_guild_monitored(guild.id):
                continue
            try:
                await self.sync_guild(guild)
            except Exception as exc:
                LOGGER.warning(
                    "Could not auto-join VC (guild_id=%s, error=%s)",
                    guild.id,
                    type(exc).__name__,
                )
                await self._record_error(guild.id, "VCへ自動参加できません")

    def _load_runtime(self) -> None:
        _ensure_opus_loaded()
        self._voice_recv = importlib.import_module("discord.ext.voice_recv")
        self._chunker = VoicePcmChunker(
            self._config.voice_chunk_seconds,
            min_rms=self._config.voice_min_rms,
            min_utterance_ms=self._config.voice_min_utterance_ms,
        )
        self._runtime_available = True

    async def _connect_guild(self, guild_id: int, channel_id: int) -> None:
        if self._voice_recv is None or self._chunker is None:
            raise RuntimeError("voice runtime is not loaded")
        channel = self._client.get_channel(channel_id)
        if channel is None:
            channel = await self._client.fetch_channel(channel_id)
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            raise RuntimeError("configured voice channel is unavailable")
        if channel.guild.id != guild_id:
            raise RuntimeError("configured voice channel is outside the target guild")

        existing = channel.guild.voice_client
        previous_channel_id = getattr(getattr(existing, "channel", None), "id", None)
        if existing is not None and (
            not hasattr(existing, "listen")
            or not self._voice_client_is_connected(existing)
        ):
            await existing.disconnect(force=True)
            self._sessions.pop(guild_id, None)
            existing = None
        if existing is not None:
            if existing.channel.id != channel.id:
                await existing.move_to(channel)
            voice_client = existing
        else:
            voice_client = await channel.connect(
                cls=self._voice_recv.VoiceRecvClient,
                reconnect=True,
                self_deaf=False,
            )

        is_listening = getattr(voice_client, "is_listening", None)
        if not callable(is_listening) or not is_listening():
            voice_client.listen(self._make_sink(guild_id))
        self._sessions[guild_id] = voice_client
        self._last_error_by_guild.pop(guild_id, None)
        LOGGER.info("VC moderation started (guild_id=%s, channel_id=%s)", guild_id, channel_id)
        if previous_channel_id != channel_id or (guild_id, channel_id) not in self._announced_channels:
            await self._announce_monitoring(guild_id, channel_id)

    async def handle_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if self._closing or not self._config.voice_enabled:
            return
        guild_id = member.guild.id
        if not self._config.is_guild_monitored(guild_id):
            return

        before_channel_id = before.channel.id if before.channel else None
        after_channel_id = after.channel.id if after.channel else None
        if before_channel_id is not None and before_channel_id != after_channel_id:
            self.forget_speaker(guild_id, member.id)

        if not self._config.voice_auto_join:
            return
        if guild_id in self._auto_join_suppressed:
            return
        client_user_id = self._client.user.id if self._client.user else None
        if member.bot and member.id != client_user_id:
            return

        preferred = after.channel if not member.bot else None
        lock = self._guild_locks.setdefault(guild_id, asyncio.Lock())
        async with lock:
            try:
                await self._sync_auto_guild(member.guild, preferred)
            except Exception as exc:
                LOGGER.warning(
                    "Could not update automatic VC monitoring (guild_id=%s, error=%s)",
                    guild_id,
                    type(exc).__name__,
                )
                await self._record_error(guild_id, "VC自動監視の更新に失敗しました")

    async def join_channel(self, guild_id: int, channel_id: int) -> None:
        if not self._config.voice_enabled or self._chunker is None:
            raise RuntimeError("VC moderation runtime is unavailable")
        if not self._config.is_guild_monitored(guild_id):
            raise RuntimeError("guild is outside the monitoring scope")
        self._auto_join_suppressed.discard(guild_id)
        lock = self._guild_locks.setdefault(guild_id, asyncio.Lock())
        async with lock:
            await self._connect_guild(guild_id, channel_id)

    async def leave_guild(self, guild_id: int) -> None:
        self._auto_join_suppressed.add(guild_id)
        guild = self._client.get_guild(guild_id)
        existing = guild.voice_client if guild is not None else None
        lock = self._guild_locks.setdefault(guild_id, asyncio.Lock())
        async with lock:
            await self._disconnect_guild(guild_id, existing)

    async def set_auto_join_for_guild(self, guild_id: int, enabled: bool) -> None:
        if not self._config.voice_auto_join:
            raise RuntimeError("automatic VC monitoring is disabled by configuration")
        if enabled:
            self._auto_join_suppressed.discard(guild_id)
            guild = self._client.get_guild(guild_id)
            if guild is not None:
                lock = self._guild_locks.setdefault(guild_id, asyncio.Lock())
                async with lock:
                    await self._sync_auto_guild(guild)
        else:
            self._auto_join_suppressed.add(guild_id)
            guild = self._client.get_guild(guild_id)
            existing = guild.voice_client if guild is not None else None
            lock = self._guild_locks.setdefault(guild_id, asyncio.Lock())
            async with lock:
                await self._disconnect_guild(guild_id, existing)

    def is_auto_join_enabled(self, guild_id: int) -> bool:
        return (
            self._config.voice_auto_join
            and guild_id not in self._auto_join_suppressed
        )

    def current_channel(self, guild_id: int) -> Optional[object]:
        guild = self._client.get_guild(guild_id)
        voice_client = guild.voice_client if guild is not None else None
        if not self._voice_client_is_connected(voice_client):
            return None
        return getattr(voice_client, "channel", None)

    @staticmethod
    def _voice_client_is_connected(voice_client: Optional[object]) -> bool:
        if voice_client is None:
            return False
        is_connected = getattr(voice_client, "is_connected", None)
        if not callable(is_connected):
            return True
        try:
            return bool(is_connected())
        except Exception:
            return False

    @property
    def runtime_available(self) -> bool:
        return self._runtime_available

    def status_summary(self, guild_id: int) -> str:
        runtime = "利用可能" if self._runtime_available else "停止"
        last_at = self._last_transcription_at.get(guild_id)
        if last_at is None:
            last_text = "まだありません"
        else:
            elapsed = max(0, int(time.monotonic() - last_at))
            last_text = f"{elapsed}秒前"
        discarded = self._rejected_chunks.get(guild_id, 0)
        if self._chunker is not None:
            discarded += self._chunker.discarded_for_guild(guild_id)
        error = self._last_error_by_guild.get(guild_id, "なし")
        return (
            f"音声受信: {runtime}\n"
            f"認識済みチャンク: {self._processed_chunks.get(guild_id, 0)}件\n"
            f"無音・低品質除外: {discarded}件\n"
            f"最終音声認識: {last_text}\n"
            f"直近エラー: {error}"
        )

    async def sync_guild(self, guild: discord.Guild) -> None:
        if (
            not self._config.voice_enabled
            or not self._config.voice_auto_join
            or guild.id in self._auto_join_suppressed
            or not self._config.is_guild_monitored(guild.id)
        ):
            return
        lock = self._guild_locks.setdefault(guild.id, asyncio.Lock())
        async with lock:
            await self._sync_auto_guild(guild)

    async def _sync_auto_guild(
        self,
        guild: discord.Guild,
        preferred: Optional[discord.abc.Connectable] = None,
    ) -> None:
        existing = guild.voice_client
        if existing is not None and not self._voice_client_is_connected(existing):
            await self._disconnect_guild(guild.id, existing)
            existing = None
        current_channel = getattr(existing, "channel", None)
        if current_channel is not None and self._human_count(current_channel) > 0:
            self._sessions[guild.id] = existing
            return

        target = None
        if preferred is not None and self._human_count(preferred) > 0:
            target = preferred
        if target is None:
            active_channels = [
                channel
                for channel in self._guild_voice_channels(guild)
                if self._human_count(channel) > 0
            ]
            if active_channels:
                target = max(active_channels, key=self._human_count)

        if target is None:
            await self._disconnect_guild(guild.id, existing)
            return
        await self._connect_guild(guild.id, target.id)

    @staticmethod
    def _guild_voice_channels(guild: discord.Guild) -> Tuple[object, ...]:
        voice_channels = tuple(getattr(guild, "voice_channels", ()))
        stage_channels = tuple(getattr(guild, "stage_channels", ()))
        return voice_channels + stage_channels

    @staticmethod
    def _human_count(channel: object) -> int:
        return sum(
            1
            for member in getattr(channel, "members", ())
            if not getattr(member, "bot", False)
        )

    async def _disconnect_guild(
        self, guild_id: int, voice_client: Optional[object] = None
    ) -> None:
        session = voice_client or self._sessions.get(guild_id)
        self._sessions.pop(guild_id, None)
        self._announced_channels = {
            key for key in self._announced_channels if key[0] != guild_id
        }
        if session is None:
            return
        if hasattr(session, "stop_listening"):
            session.stop_listening()
        await session.disconnect(force=True)
        LOGGER.info("VC moderation stopped because no human users remain (guild_id=%s)", guild_id)

    def _make_sink(self, guild_id: int) -> object:
        manager = self
        voice_recv = self._voice_recv
        loop = asyncio.get_running_loop()

        class ModerationSink(voice_recv.AudioSink):
            def wants_opus(self) -> bool:
                return False

            def write(self, user: object, data: object) -> None:
                user_id = getattr(user, "id", None)
                if user_id is None or getattr(user, "bot", False):
                    return
                try:
                    chunks = manager._chunker.accept_pcm(
                        guild_id, user_id, data.pcm
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Could not process VC audio (guild_id=%s, error=%s)",
                        guild_id,
                        type(exc).__name__,
                    )
                    return
                for wav_audio in chunks:
                    asyncio.run_coroutine_threadsafe(
                        manager._process_audio_chunk(guild_id, user_id, wav_audio),
                        loop,
                    )

            def cleanup(self) -> None:
                return None

            def on_voice_member_speaking_stop(self, member: object) -> None:
                user_id = getattr(member, "id", None)
                if user_id is None or getattr(member, "bot", False):
                    return
                wav_audio = manager._chunker.flush(guild_id, user_id)
                if wav_audio:
                    asyncio.run_coroutine_threadsafe(
                        manager._process_audio_chunk(guild_id, user_id, wav_audio),
                        loop,
                    )

        return ModerationSink()

    def forget_speaker(self, guild_id: int, user_id: int) -> None:
        if self._chunker is not None:
            wav_audio = self._chunker.flush(guild_id, user_id)
            if wav_audio:
                asyncio.create_task(
                    self._process_audio_chunk(guild_id, user_id, wav_audio)
                )

    async def _process_audio_chunk(
        self, guild_id: int, user_id: int, wav_audio: bytes
    ) -> None:
        try:
            key = (guild_id, user_id)
            lock = self._speaker_locks.setdefault(key, asyncio.Lock())
            async with lock:
                transcript = await self._service.transcribe_wav(wav_audio)
                if not transcript:
                    self._rejected_chunks[guild_id] += 1
                    LOGGER.info(
                        "VC audio chunk rejected by speech quality checks (guild_id=%s)",
                        guild_id,
                    )
                    return
                self._processed_chunks[guild_id] += 1
                self._last_transcription_at[guild_id] = time.monotonic()
                self._last_error_by_guild.pop(guild_id, None)
                alerts_sent = await self._moderator.process(
                    guild_id, user_id, transcript
                )
                LOGGER.info(
                    "VC transcription analyzed (guild_id=%s, alerts_sent=%s)",
                    guild_id,
                    alerts_sent,
                )
        except Exception as exc:
            LOGGER.warning(
                "Could not send VC moderation alert (guild_id=%s, error=%s)",
                guild_id,
                type(exc).__name__,
            )
            await self._record_error(guild_id, "音声認識または通知に失敗しました")

    async def _announce_monitoring(self, guild_id: int, channel_id: int) -> None:
        key = (guild_id, channel_id)
        if key in self._announced_channels:
            return
        self._announced_channels.add(key)
        if self._notice_sender is None:
            return
        try:
            await self._notice_sender(
                guild_id,
                channel_id,
                "🎙️ このVCではモデレーション判定のため音声を短時間だけ認識します。"
                "音声と文字起こしはBotのファイルへ保存しません。"
                "管理者は `/vc_leave` で停止できます。",
            )
        except Exception as exc:
            LOGGER.warning(
                "Could not send VC monitoring notice (guild_id=%s, error=%s)",
                guild_id,
                type(exc).__name__,
            )

    async def _record_error(self, guild_id: int, message: str) -> None:
        self._last_error_by_guild[guild_id] = message
        if self._notice_sender is None:
            return
        current = time.monotonic()
        previous = self._last_error_notice_at.get(guild_id)
        if previous is not None and current - previous < 300:
            return
        self._last_error_notice_at[guild_id] = current
        try:
            await self._notice_sender(
                guild_id,
                None,
                f"⚠️ {message}。テキスト監視は継続しています。"
                "詳しくは `/vc_status` と `/bot_status` を確認してください。",
            )
        except Exception as exc:
            LOGGER.warning(
                "Could not send VC health notice (guild_id=%s, error=%s)",
                guild_id,
                type(exc).__name__,
            )

    async def close(self) -> None:
        self._closing = True
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        for voice_client in sessions:
            try:
                if hasattr(voice_client, "stop_listening"):
                    voice_client.stop_listening()
                await voice_client.disconnect(force=True)
            except Exception as exc:
                LOGGER.warning("Could not close VC session (error=%s)", type(exc).__name__)
        if self._chunker is not None:
            self._chunker.clear()
        self._runtime_available = False
