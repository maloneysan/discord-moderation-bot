from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Mapping, Optional


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class BotConfig:
    token: str
    guild_ids: FrozenSet[int]
    monitored_channel_ids: FrozenSet[int]
    alert_channel_ids: Mapping[int, int]
    rules_path: Path
    monitor_all_guilds: bool = False
    voice_enabled: bool = False
    voice_auto_join: bool = False
    voice_channel_ids: Mapping[int, int] = field(default_factory=dict)
    voice_alert_channel_ids: Mapping[int, int] = field(default_factory=dict)
    moderation_backend: str = "local"
    groq_api_key: Optional[str] = None
    groq_text_model: str = "openai/gpt-oss-120b"
    groq_fallback_text_model: str = "openai/gpt-oss-20b"
    groq_voice_text_model: str = "openai/gpt-oss-safeguard-20b"
    groq_speech_model: str = "whisper-large-v3-turbo"
    groq_fallback_speech_model: str = "whisper-large-v3"
    groq_confidence_threshold: int = 25
    groq_timeout_seconds: float = 20.0
    groq_text_interval_seconds: float = 10.0
    groq_voice_interval_seconds: float = 10.0
    groq_audio_interval_seconds: float = 6.0
    voice_chunk_seconds: int = 10
    context_message_count: int = 3
    context_ttl_seconds: int = 180
    voice_min_rms: int = 25
    voice_min_utterance_ms: int = 100
    runtime_state_path: Optional[Path] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "BotConfig":
        values = os.environ if env is None else env

        token = values.get("DISCORD_BOT_TOKEN", "").strip()
        if not token:
            raise ConfigError("DISCORD_BOT_TOKEN is required")

        monitor_all_guilds = _parse_bool(
            values.get("MONITOR_ALL_GUILDS", "false"), "MONITOR_ALL_GUILDS"
        )
        guild_ids = _parse_guild_ids(values, required=not monitor_all_guilds)
        monitored = _parse_id_list(
            values.get("MONITORED_CHANNEL_IDS", ""), "MONITORED_CHANNEL_IDS"
        )
        alert_channel_ids = _parse_alert_channels(
            values, guild_ids, monitor_all_guilds
        )
        voice_channel_ids = _parse_id_mapping(
            values.get("VOICE_CHANNEL_IDS", ""), "VOICE_CHANNEL_IDS"
        )
        _validate_mapping_guilds(
            voice_channel_ids,
            guild_ids,
            "VOICE_CHANNEL_IDS",
            allow_all_guilds=monitor_all_guilds,
        )

        voice_alert_channel_ids = _parse_id_mapping(
            values.get("VOICE_ALERT_CHANNEL_IDS", ""),
            "VOICE_ALERT_CHANNEL_IDS",
        )
        _validate_mapping_guilds(
            voice_alert_channel_ids,
            guild_ids,
            "VOICE_ALERT_CHANNEL_IDS",
            allow_all_guilds=monitor_all_guilds,
        )

        voice_enabled = _parse_bool(values.get("VOICE_ENABLED", "false"), "VOICE_ENABLED")
        voice_auto_join = _parse_bool(
            values.get("VOICE_AUTO_JOIN", "false"), "VOICE_AUTO_JOIN"
        )
        if voice_auto_join and not voice_enabled:
            raise ConfigError("VOICE_AUTO_JOIN requires VOICE_ENABLED=true")
        moderation_backend = values.get("MODERATION_BACKEND", "local").strip().casefold()
        if moderation_backend not in {"local", "groq"}:
            raise ConfigError("MODERATION_BACKEND must be local or groq")
        groq_api_key = values.get("GROQ_API_KEY", "").strip() or None
        if moderation_backend == "groq" and groq_api_key is None:
            raise ConfigError("GROQ_API_KEY is required when MODERATION_BACKEND=groq")
        groq_text_model = (
            values.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b").strip()
            or "openai/gpt-oss-120b"
        )
        groq_fallback_text_model = (
            values.get("GROQ_FALLBACK_TEXT_MODEL", "openai/gpt-oss-20b").strip()
            or "openai/gpt-oss-20b"
        )
        groq_voice_text_model = (
            values.get(
                "GROQ_VOICE_TEXT_MODEL", "openai/gpt-oss-safeguard-20b"
            ).strip()
            or "openai/gpt-oss-safeguard-20b"
        )
        groq_speech_model = (
            values.get("GROQ_SPEECH_MODEL", "whisper-large-v3-turbo").strip()
            or "whisper-large-v3-turbo"
        )
        groq_fallback_speech_model = (
            values.get(
                "GROQ_FALLBACK_SPEECH_MODEL", "whisper-large-v3"
            ).strip()
            or "whisper-large-v3"
        )
        groq_confidence_threshold = _parse_bounded_int(
            values.get("GROQ_CONFIDENCE_THRESHOLD", "25"),
            "GROQ_CONFIDENCE_THRESHOLD",
            0,
            100,
        )
        groq_timeout_seconds = _parse_positive_float(
            values.get("GROQ_TIMEOUT_SECONDS", "20"), "GROQ_TIMEOUT_SECONDS"
        )
        groq_text_interval_seconds = _parse_positive_float(
            values.get("GROQ_TEXT_INTERVAL_SECONDS", "10"),
            "GROQ_TEXT_INTERVAL_SECONDS",
        )
        groq_voice_interval_seconds = _parse_positive_float(
            values.get("GROQ_VOICE_INTERVAL_SECONDS", "10"),
            "GROQ_VOICE_INTERVAL_SECONDS",
        )
        groq_audio_interval_seconds = _parse_positive_float(
            values.get("GROQ_AUDIO_INTERVAL_SECONDS", "6"),
            "GROQ_AUDIO_INTERVAL_SECONDS",
        )
        voice_chunk_seconds = _parse_bounded_int(
            values.get("VOICE_CHUNK_SECONDS", "10"),
            "VOICE_CHUNK_SECONDS",
            5,
            30,
        )
        context_message_count = _parse_bounded_int(
            values.get("CONTEXT_MESSAGE_COUNT", "3"),
            "CONTEXT_MESSAGE_COUNT",
            1,
            5,
        )
        context_ttl_seconds = _parse_bounded_int(
            values.get("CONTEXT_TTL_SECONDS", "180"),
            "CONTEXT_TTL_SECONDS",
            30,
            600,
        )
        voice_min_rms = _parse_bounded_int(
            values.get("VOICE_MIN_RMS", "25"),
            "VOICE_MIN_RMS",
            0,
            5000,
        )
        voice_min_utterance_ms = _parse_bounded_int(
            values.get("VOICE_MIN_UTTERANCE_MS", "100"),
            "VOICE_MIN_UTTERANCE_MS",
            100,
            2000,
        )
        if voice_enabled:
            if not voice_auto_join and not voice_channel_ids:
                raise ConfigError("VOICE_CHANNEL_IDS is required when VOICE_ENABLED is true")
            if moderation_backend != "groq":
                raise ConfigError("VOICE_ENABLED requires MODERATION_BACKEND=groq")
            missing_alerts = (
                set(voice_channel_ids)
                - set(voice_alert_channel_ids)
                - set(alert_channel_ids)
            )
            if missing_alerts and not voice_auto_join:
                raise ConfigError(
                    "VOICE_ALERT_CHANNEL_IDS or ALERT_CHANNEL_IDS is required for every voice guild"
                )

        rules_path = Path(__file__).resolve().parent.parent / "config" / "rules.json"
        return cls(
            token=token,
            guild_ids=frozenset(guild_ids),
            monitored_channel_ids=frozenset(monitored),
            alert_channel_ids=dict(alert_channel_ids),
            rules_path=rules_path,
            monitor_all_guilds=monitor_all_guilds,
            voice_enabled=voice_enabled,
            voice_auto_join=voice_auto_join,
            voice_channel_ids=dict(voice_channel_ids),
            voice_alert_channel_ids=dict(voice_alert_channel_ids),
            moderation_backend=moderation_backend,
            groq_api_key=groq_api_key,
            groq_text_model=groq_text_model,
            groq_fallback_text_model=groq_fallback_text_model,
            groq_voice_text_model=groq_voice_text_model,
            groq_speech_model=groq_speech_model,
            groq_fallback_speech_model=groq_fallback_speech_model,
            groq_confidence_threshold=groq_confidence_threshold,
            groq_timeout_seconds=groq_timeout_seconds,
            groq_text_interval_seconds=groq_text_interval_seconds,
            groq_voice_interval_seconds=groq_voice_interval_seconds,
            groq_audio_interval_seconds=groq_audio_interval_seconds,
            voice_chunk_seconds=voice_chunk_seconds,
            context_message_count=context_message_count,
            context_ttl_seconds=context_ttl_seconds,
            voice_min_rms=voice_min_rms,
            voice_min_utterance_ms=voice_min_utterance_ms,
            runtime_state_path=rules_path.parent / "runtime_alert_channels.json",
        )

    def alert_channel_for(self, guild_id: int) -> Optional[int]:
        return self.alert_channel_ids.get(guild_id)

    def voice_alert_channel_for(self, guild_id: int) -> Optional[int]:
        return self.voice_alert_channel_ids.get(
            guild_id, self.alert_channel_ids.get(guild_id)
        )

    def is_guild_monitored(self, guild_id: int) -> bool:
        return self.monitor_all_guilds or guild_id in self.guild_ids


def _parse_guild_ids(
    values: Mapping[str, str], required: bool = True
) -> FrozenSet[int]:
    raw_multiple = values.get("DISCORD_GUILD_IDS", "").strip()
    raw_legacy = values.get("DISCORD_GUILD_ID", "").strip()
    if raw_multiple:
        return _parse_id_list(raw_multiple, "DISCORD_GUILD_IDS")
    if raw_legacy:
        return frozenset({_parse_required_id(raw_legacy, "DISCORD_GUILD_ID")})
    if required:
        raise ConfigError("DISCORD_GUILD_IDS is required")
    return frozenset()


def _parse_alert_channels(
    values: Mapping[str, str], guild_ids: FrozenSet[int], monitor_all_guilds: bool
) -> Mapping[int, int]:
    mapped = _parse_id_mapping(
        values.get("ALERT_CHANNEL_IDS", ""), "ALERT_CHANNEL_IDS"
    )
    _validate_mapping_guilds(
        mapped,
        guild_ids,
        "ALERT_CHANNEL_IDS",
        allow_all_guilds=monitor_all_guilds,
    )

    raw_legacy = values.get("ALERT_CHANNEL_ID", "").strip()
    if raw_legacy:
        if len(guild_ids) != 1:
            raise ConfigError(
                "ALERT_CHANNEL_ID can only be used with one guild; use ALERT_CHANNEL_IDS"
            )
        mapped = dict(mapped)
        mapped[next(iter(guild_ids))] = _parse_required_id(
            raw_legacy, "ALERT_CHANNEL_ID"
        )
    return mapped


def _parse_required_id(value: Optional[str], name: str) -> int:
    raw = "" if value is None else value.strip()
    if not raw:
        raise ConfigError(f"{name} is required")
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return parsed


def _parse_id_list(value: str, name: str) -> FrozenSet[int]:
    if not value.strip():
        return frozenset()

    parsed = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parsed.add(_parse_required_id(item, name))
    return frozenset(parsed)


def _parse_id_mapping(value: str, name: str) -> Mapping[int, int]:
    if not value.strip():
        return {}

    parsed = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split(":", 1)
        if len(parts) != 2:
            raise ConfigError(f"{name} must use guild_id:channel_id entries")
        guild_id = _parse_required_id(parts[0], name)
        channel_id = _parse_required_id(parts[1], name)
        parsed[guild_id] = channel_id
    return parsed


def _validate_mapping_guilds(
    mapping: Mapping[int, int],
    guild_ids: FrozenSet[int],
    name: str,
    allow_all_guilds: bool = False,
) -> None:
    if allow_all_guilds:
        return
    unknown = set(mapping) - set(guild_ids)
    if unknown:
        raise ConfigError(f"{name} contains a guild not listed in DISCORD_GUILD_IDS")


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(f"{name} must be true or false")


def _parse_bounded_int(value: str, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _parse_positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return parsed
