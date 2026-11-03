"""In-process rate limiting and concurrency limits.

These protect a single API process from brute-force logins and from
exhausting the local language model. They are per-process: a multi-process
or multi-host deployment needs a shared limiter (for example at a reverse
proxy or in Redis) in addition to these.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


class RateLimiter:
    """Allow at most `limit` events per key within a sliding time window."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record an event. Return seconds to wait if the key is over its limit."""
        now = self._clock()
        cutoff = now - self.window
        with self._lock:
            events = self._events.get(key)
            if events is None:
                if len(self._events) >= self._max_keys:
                    self._prune(cutoff)
                events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return max(events[0] + self.window - now, 0.0)
            events.append(now)
            return None

    def _prune(self, cutoff: float) -> None:
        for key in [k for k, events in self._events.items() if not events or events[-1] <= cutoff]:
            del self._events[key]
        if len(self._events) >= self._max_keys:
            # Under a flood of distinct keys, forget the oldest rather than grow.
            oldest = sorted(self._events, key=lambda k: self._events[k][-1])
            for key in oldest[: len(oldest) // 2]:
                del self._events[key]

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


@dataclass
class Lease:
    """A held concurrency slot. Releasing twice is harmless."""

    _release: Callable[[], None]
    _released: bool = field(default=False)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._release()


class ConcurrencyGate:
    """Non-blocking limit on simultaneous work, overall and per user."""

    def __init__(self, total: int, per_user: int) -> None:
        self._total = total
        self._per_user = per_user
        self._active: dict[str, int] = {}
        self._lock = threading.Lock()

    def try_acquire(self, user_id: str) -> Lease | None:
        with self._lock:
            if sum(self._active.values()) >= self._total:
                return None
            if self._active.get(user_id, 0) >= self._per_user:
                return None
            self._active[user_id] = self._active.get(user_id, 0) + 1
        return Lease(lambda: self._release(user_id))

    def _release(self, user_id: str) -> None:
        with self._lock:
            remaining = self._active.get(user_id, 0) - 1
            if remaining > 0:
                self._active[user_id] = remaining
            else:
                self._active.pop(user_id, None)

    @property
    def active(self) -> int:
        with self._lock:
            return sum(self._active.values())
