"""Shared HTTP plumbing: retry/backoff, polite rate limiting, disk cache, offline mode.

Design notes
------------
* Every public endpoint used by this project is free and key-less (or free-key,
  e.g. FRED). Nothing here requires a paid subscription.
* ``offline=True`` makes the pipeline strictly cache-read: this is what CI smoke
  tests and reproducible backfill runs use, and it guarantees that a network
  hiccup degrades a *pillar* rather than the whole run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# SEC requires a declared contact in the User-Agent and answers 403 to generic
# bot strings (reproduced: same URL, same IP, only the UA differs). Every
# SEC-backed endpoint shares this constant; ``SEC_USER_AGENT`` overrides it.
DEFAULT_CONTACT_EMAIL = "rishichowdhury080@gmail.com"


def compliant_user_agent() -> str:
    """A polite, identifiable User-Agent for every government/free data source."""
    return os.getenv("SEC_USER_AGENT", "").strip() or f"AutonomousPMS/4.0 (contact: {DEFAULT_CONTACT_EMAIL})"


DEFAULT_UA = compliant_user_agent()


class SourceUnavailable(RuntimeError):
    """Raised when a source cannot be reached and no cached copy exists."""


class PacingExhausted(RuntimeError):
    """The server asked for a wait longer than any sensible step budget.

    Raised instead of sleeping (e.g. OpenAlex answers exhausted shared-IP credits
    with ``Retry-After ≈ 4 hours``); the caller degrades this source gracefully.
    """


class HttpClient:
    """httpx wrapper with retry, rate limiting, caching and an offline switch."""

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 25.0,
        attempts: int = 3,
        min_interval: float = 0.34,
        retry_after_floor: float = 0.0,
        retry_after_cap: float = 60.0,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: float | None = 86_400.0,
        offline: bool = False,
    ) -> None:
        merged = {"User-Agent": DEFAULT_UA, "Accept": "application/json"}
        merged.update(headers or {})
        self.headers = merged
        self.timeout = timeout
        self.attempts = attempts
        self.min_interval = min_interval
        # Minimum extra wait applied after a 429 response *without* a Retry-After
        # header. Sources with a documented pacing (e.g. GDELT: one call per 5 s)
        # pass their interval here so a 429 can never be met with a sub-interval
        # retry; the wait and the next-request throttle are both raised to it.
        self.retry_after_floor = max(0.0, float(retry_after_floor))
        # A server may advertise a Retry-After longer than any step budget (OpenAlex
        # answers exhausted shared-IP credits with Retry-After ≈ 4 hours). Honouring
        # that verbatim would sleep through the job timeout, so waits above this cap
        # give up immediately and let the caller degrade that one source/company.
        self.retry_after_cap = max(1.0, float(retry_after_cap))
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.offline = offline
        self._client = httpx.Client(headers=self.headers, timeout=timeout, follow_redirects=True)
        self._last_call = 0.0
        self._retry_not_before = 0.0
        self.stats = {"requests": 0, "cache_hits": 0, "failures": 0}

    # ------------------------------------------------------------------ public --
    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        # Some free APIs answer rate limits with HTTP 200 and a *plain-text* notice
        # (observed on GDELT). json.loads would then raise a bare JSONDecodeError that
        # bypasses every ``except SourceUnavailable`` guard and kills the pillar.
        key = self._cache_key("GET", url, params)
        text = self.get_text(url, params=params, extra_headers=extra_headers)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # Purge so the poisoned cache entry cannot shadow the retry window.
            self._purge_cache(key)
            head = " ".join(text[:120].split())
            raise SourceUnavailable(f"GET {url} returned non-JSON payload: {head!r}") from exc

    def post_json(self, url: str, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> Any:
        key = self._cache_key("POST", url, payload)
        cached = self._read_cache(key)
        if cached is not None:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                # Legacy poisoned entry (cached before the JSON guard); refetch.
                self._purge_cache(key)
        if self.offline:
            raise SourceUnavailable(f"offline and no cache for POST {url}")
        text = self._request("POST", url, json_body=payload, extra_headers=extra_headers)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            self._purge_cache(key)
            head = " ".join(text[:120].split())
            raise SourceUnavailable(f"POST {url} returned non-JSON payload: {head!r}") from exc
        self._write_cache(key, text)
        return parsed

    def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        key = self._cache_key("GET", url, params)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        if self.offline:
            raise SourceUnavailable(f"offline and no cache for GET {url}")
        text = self._request("GET", url, params=params, extra_headers=extra_headers)
        self._write_cache(key, text)
        return text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- internals
    def _throttle(self) -> None:
        # A 429 (or a Retry-After header) makes the *next* call wait longer than the
        # steady-state pacing; honour it before the regular interval gate.
        wait = max(self.min_interval, self._retry_not_before - time.monotonic())
        if wait <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < wait:
            time.sleep(wait - elapsed)
        self._last_call = time.monotonic()

    def _retry_wait(self, retry_state: Any) -> float:
        """Exponential backoff, raised to any server-advertised Retry-After delay."""
        try:
            base = wait_exponential(multiplier=1, min=2, max=10)(retry_state)
        except Exception:  # pragma: no cover - tenacity API drift guard
            base = 2.0
        return max(float(base or 0.0), self._retry_not_before - time.monotonic(), 0.0)

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        @retry(
            stop=stop_after_attempt(self.attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.HTTPError)),
            reraise=True,
        )
        def _call() -> str:
            self._throttle()
            self.stats["requests"] += 1
            if method == "POST":
                resp = self._client.post(url, json=json_body, headers=extra_headers)
            else:
                resp = self._client.get(url, params=params, headers=extra_headers)
            if resp.status_code == 429:
                seconds = self.retry_after_floor
                header = resp.headers.get("Retry-After", "").strip()
                try:
                    seconds = max(seconds, float(header))
                except ValueError:
                    pass
                if seconds > self.retry_after_cap:
                    # Not retryable: sleeping hours inside a step is worse than
                    # degrading one source for this run.
                    raise PacingExhausted(
                        f"{url} asked to wait {seconds:.0f}s (> cap {self.retry_after_cap:.0f}s)"
                    )
                self._retry_not_before = max(self._retry_not_before, time.monotonic() + seconds)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.text

        try:
            return _call()
        except Exception as exc:  # pragma: no cover - network dependent
            self.stats["failures"] += 1
            raise SourceUnavailable(f"{method} {url} failed: {exc}") from exc

    def _cache_key(self, method: str, url: str, payload: Any) -> str:
        blob = json.dumps({"m": method, "u": url, "p": payload}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        bucket = self.cache_dir / key[:2]
        bucket.mkdir(parents=True, exist_ok=True)
        return bucket / f"{key}.txt"

    def _read_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        if self.cache_ttl_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > self.cache_ttl_seconds and not self.offline:
                return None
        self.stats["cache_hits"] += 1
        return path.read_text(encoding="utf-8")

    def _write_cache(self, key: str, text: str) -> None:
        path = self._cache_path(key)
        if path is not None:
            path.write_text(text, encoding="utf-8")

    def _purge_cache(self, key: str) -> None:
        path = self._cache_path(key)
        if path is not None:
            path.unlink(missing_ok=True)


def safe_call(fn: Callable[..., Any], *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Run a fetcher; on failure return ``default`` so one dead source cannot kill a run."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def write_shard(frame: pd.DataFrame, output_dir: str | Path, name: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    payload = frame if frame is not None else pd.DataFrame()
    if payload.shape[1] == 0:
        # A column-less frame cannot be written to Parquet; keep a marker column so the
        # downstream Fan-In can see "this shard ran and produced nothing" instead of
        # failing the whole job.
        payload = pd.DataFrame({"_empty": pd.Series(dtype="object")})
    payload.to_parquet(path, compression="zstd", index=False)
    return path
