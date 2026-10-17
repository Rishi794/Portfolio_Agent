#!/usr/bin/env python3
"""Cron-slot gate for the scheduled pipeline (Stage 1, "prepare").

Decides whether the *current* GitHub schedule tick should run the parallel DAG:

* Non-session days (weekends, NSE/BSE and NYSE holidays) -> ``run=false`` so no
  worker spins up at all.
* The US pipeline is scheduled twice (``45 20`` UTC for EDT and ``45 21`` UTC for
  EST). Exactly one of the two slots lands ~45 minutes after the actual 4:00 PM
  ET close; the gate compares ``now`` with the exchange calendar's DST-aware
  close and skips the other slot.
* India (``30 10`` UTC) lands 30 minutes after the 3:30 PM IST close.

Emits ``markets`` (expanded list, e.g. ``US,INDIA`` / ``US``) and ``run`` to
``$GITHUB_OUTPUT`` (stdout when run locally). Usage:

    uv run --no-project --with exchange_calendars python scripts/market_open_gate.py
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
from exchange_calendars.errors import DateOutOfBounds

CALENDARS = {"INDIA": "XBOM", "US": "XNYS"}

# Local close used only once the pinned calendar range ends (see _calendar()).
LOCAL_CLOSES = {"INDIA": (15, 30, "Asia/Kolkata"), "US": (16, 0, "America/New_York")}

# A slot fires only inside (close, close + 90m]. India ticks 30m after the close,
# the US ticks 45m after it, and the *wrong* DST slot is ~1h45m out -> skipped.
MIN_AFTER_CLOSE = dt.timedelta(0)
MAX_AFTER_CLOSE = dt.timedelta(minutes=90)

_CALENDAR_CACHE: dict[str, object] = {}


def _calendar(market: str):
    if market not in _CALENDAR_CACHE:
        name = CALENDARS[market]
        try:
            # Widen the default range where the calendar is rule-based (XNYS).
            # XBOM publishes holiday tables per year and rejects the extension,
            # in which case we fall back to the pinned range.
            _CALENDAR_CACHE[market] = xcals.get_calendar(name, start="2016-01-01", end="2030-12-31")
        except ValueError:
            _CALENDAR_CACHE[market] = xcals.get_calendar(name)
    return _CALENDAR_CACHE[market]


def _fallback_close(market: str, day: dt.date) -> dt.datetime:
    hour, minute, tzname = LOCAL_CLOSES[market]
    local = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=ZoneInfo(tzname))
    return local.astimezone(dt.timezone.utc)


def _emit(name: str, value: str) -> None:
    handle_path = os.getenv("GITHUB_OUTPUT")
    if handle_path:
        with Path(handle_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    print(f"{name}={value}")


def market_is_open_now(market: str, now: dt.datetime) -> bool:
    """True when today is an exchange session and `now` sits just after its close."""
    calendar = _calendar(market)
    today = pd.Timestamp(now.date())
    close: dt.datetime | None
    try:
        if calendar.is_session(today):
            close = calendar.session_close(today).to_pydatetime()
        else:
            close = None
    except DateOutOfBounds:
        # Beyond the calendar's pinned holiday table (XBOM stops at 2026): keep
        # the schedule alive with a weekday + DST-aware local-close heuristic.
        if now.weekday() >= 5:
            return False
        close = _fallback_close(market, now.date())
    if close is None:
        return False
    return MIN_AFTER_CLOSE <= (now - close) <= MAX_AFTER_CLOSE


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    event = os.getenv("GITHUB_EVENT_NAME", "")
    target = (os.getenv("TARGET_MARKET") or "BOTH").upper()

    candidates = [market for market in ("INDIA", "US") if target in ("BOTH", market)]
    if event == "workflow_dispatch":
        # Manual runs ignore the clock; the operator decides.
        _emit("markets", ",".join(candidates))
        _emit("run", "true")
        return 0

    open_markets = [market for market in candidates if market_is_open_now(market, now)]
    _emit("markets", ",".join(open_markets))
    _emit("run", "true" if open_markets else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
