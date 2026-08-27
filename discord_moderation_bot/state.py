from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Tuple


class AlertRegistry:
    """Bounded in-memory claims for message/category alert de-duplication."""

    def __init__(self, max_entries: int = 10_000) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._claims: "OrderedDict[Tuple[int, str], None]" = OrderedDict()

    def claim_new(self, message_id: int, categories: Iterable[str]) -> Tuple[str, ...]:
        claimed = []
        for category in categories:
            key = (message_id, category)
            if key in self._claims:
                self._claims.move_to_end(key)
                continue
            self._claims[key] = None
            claimed.append(category)
            while len(self._claims) > self._max_entries:
                self._claims.popitem(last=False)
        return tuple(claimed)

    def release(self, message_id: int, categories: Iterable[str]) -> None:
        for category in categories:
            self._claims.pop((message_id, category), None)
