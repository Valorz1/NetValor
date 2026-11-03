from __future__ import annotations

from app.ratelimit import ConcurrencyGate, RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_allows_up_to_limit_then_reports_wait() -> None:
    clock = Clock()
    limiter = RateLimiter(3, 60, clock=clock)

    assert [limiter.hit("ip") for _ in range(3)] == [None, None, None]
    retry = limiter.hit("ip")
    assert retry is not None
    assert 59 <= retry <= 60
    assert limiter.hit("other") is None


def test_rate_limiter_window_slides() -> None:
    clock = Clock()
    limiter = RateLimiter(2, 60, clock=clock)
    limiter.hit("k")
    clock.now += 30
    limiter.hit("k")
    assert limiter.hit("k") is not None

    clock.now += 31  # the first hit has left the window
    assert limiter.hit("k") is None


def test_rate_limiter_bounds_memory_under_many_keys() -> None:
    clock = Clock()
    limiter = RateLimiter(1, 60, clock=clock, max_keys=100)
    for index in range(1000):
        limiter.hit(f"key-{index}")

    assert len(limiter._events) <= 100


def test_concurrency_gate_limits_total_and_per_user() -> None:
    gate = ConcurrencyGate(total=2, per_user=1)

    first = gate.try_acquire("sam")
    assert first is not None
    assert gate.try_acquire("sam") is None  # per-user limit
    second = gate.try_acquire("alex")
    assert second is not None
    assert gate.try_acquire("kim") is None  # total limit

    first.release()
    first.release()  # idempotent
    assert gate.active == 1
    assert gate.try_acquire("kim") is not None
