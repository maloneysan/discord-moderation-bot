from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Mapping, Optional


LOGGER = logging.getLogger(__name__)


class AlertChannelStore:
    """Persist only guild-to-channel IDs; never message or transcript content."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._channels: Dict[int, int] = {}
        self._load()

    def get(self, guild_id: int) -> Optional[int]:
        return self._channels.get(guild_id)

    def set(self, guild_id: int, channel_id: int) -> None:
        if guild_id <= 0 or channel_id <= 0:
            raise ValueError("guild and channel IDs must be positive")
        updated = dict(self._channels)
        updated[guild_id] = channel_id
        self._persist(updated)
        self._channels = updated

    def clear(self, guild_id: int) -> bool:
        if guild_id not in self._channels:
            return False
        updated = dict(self._channels)
        del updated[guild_id]
        self._persist(updated)
        self._channels = updated
        return True

    def snapshot(self) -> Mapping[int, int]:
        return dict(self._channels)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid alert channel state")
            channels = payload.get("channels", {})
            if not isinstance(channels, dict):
                raise ValueError("invalid alert channel state")
            loaded = {}
            for guild_id, channel_id in channels.items():
                parsed_guild = int(guild_id)
                if parsed_guild <= 0 or not isinstance(channel_id, int) or channel_id <= 0:
                    raise ValueError("invalid alert channel ID")
                loaded[parsed_guild] = channel_id
            self._channels = loaded
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "Could not load alert channel state; defaults used (error=%s)",
                type(exc).__name__,
            )

    def _persist(self, channels: Mapping[int, int]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {
            "version": 1,
            "channels": {
                str(guild_id): channel_id
                for guild_id, channel_id in sorted(channels.items())
            },
        }
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, self._path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
