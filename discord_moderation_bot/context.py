from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class _ContextEntry:
    message_id: int
    text: str
    created_at: float


class ConversationContextBuffer:
    """Small, expiring, memory-only channel context without author identities."""

    def __init__(
        self,
        max_messages: int = 3,
        ttl_seconds: float = 180.0,
        max_chars_per_message: int = 500,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_chars_per_message <= 0:
            raise ValueError("max_chars_per_message must be positive")
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds
        self._max_chars = max_chars_per_message
        self._channels: Dict[int, Deque[_ContextEntry]] = {}

    def recent(
        self,
        channel_id: int,
        *,
        excluding_message_id: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Tuple[str, ...]:
        current = time.monotonic() if now is None else now
        entries = self._prune(channel_id, current)
        return tuple(
            entry.text
            for entry in entries
            if entry.message_id != excluding_message_id
        )[-self._max_messages :]

    def remember(
        self,
        channel_id: int,
        message_id: int,
        text: str,
        *,
        now: Optional[float] = None,
    ) -> None:
        normalized = " ".join(text.split()).strip()[: self._max_chars]
        if not normalized:
            return
        current = time.monotonic() if now is None else now
        entries = self._prune(channel_id, current)
        retained = deque(
            (entry for entry in entries if entry.message_id != message_id),
            maxlen=self._max_messages,
        )
        retained.append(_ContextEntry(message_id, normalized, current))
        self._channels[channel_id] = retained

    @property
    def message_count(self) -> int:
        current = time.monotonic()
        for channel_id in tuple(self._channels):
            self._prune(channel_id, current)
        return sum(len(entries) for entries in self._channels.values())

    def clear(self) -> None:
        self._channels.clear()

    def _prune(self, channel_id: int, now: float) -> Deque[_ContextEntry]:
        entries = self._channels.get(channel_id, deque(maxlen=self._max_messages))
        cutoff = now - self._ttl_seconds
        retained = deque(
            (entry for entry in entries if entry.created_at >= cutoff),
            maxlen=self._max_messages,
        )
        if retained:
            self._channels[channel_id] = retained
        else:
            self._channels.pop(channel_id, None)
        return retained
