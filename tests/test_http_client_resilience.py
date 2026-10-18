"""HttpClient resilience behaviours (local mock server, no external network).

Covers the three failure modes that used to kill ingestion steps:

* a *plain-text* HTTP 200 body (GDELT rate-limit notice) must raise
  ``SourceUnavailable`` — never a bare ``JSONDecodeError`` that bypasses callers;
* a 429 whose ``Retry-After`` exceeds the cap must give up immediately instead of
  sleeping (OpenAlex answers exhausted shared-IP credits with ~4 hours);
* a short 429 must be retried after the advertised wait and then succeed.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.ingestion.base import HttpClient, SourceUnavailable


class _PlannedHandler(BaseHTTPRequestHandler):
    """Serves a scripted plan of (status, headers, body) responses per request."""

    plan: list[tuple[int, dict[str, str], bytes]] = []
    calls: int = 0

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        index = min(self.__class__.calls, len(self.__class__.plan) - 1)
        status, headers, body = self.__class__.plan[index]
        self.__class__.calls += 1
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence
        return


@pytest.fixture()
def mock_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PlannedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _PlannedHandler.calls = 0
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/endpoint"
    finally:
        server.shutdown()
        server.server_close()


def test_plaintext_200_raises_source_unavailable(mock_endpoint, tmp_path):
    # GDELT answers bursts with HTTP 200 and a plain-text rate-limit notice.
    _PlannedHandler.plan = [(200, {"Content-Type": "text/plain"}, b"Please limit requests to one every 5 seconds.")]
    client = HttpClient(cache_dir=tmp_path, attempts=1)
    with pytest.raises(SourceUnavailable, match="non-JSON"):
        client.get_json(mock_endpoint)


def test_long_retry_after_gives_up_without_sleeping(mock_endpoint, tmp_path):
    # OpenAlex: Retry-After is hours; the client must not sleep through job budgets.
    _PlannedHandler.plan = [(429, {"Retry-After": "15104"}, b"rate limited")]
    client = HttpClient(cache_dir=tmp_path, attempts=3, retry_after_cap=60.0)
    started = time.monotonic()
    with pytest.raises(SourceUnavailable, match="failed"):
        client.get_json(mock_endpoint)
    assert time.monotonic() - started < 5.0, "cap must prevent multi-hour sleeps"
    assert _PlannedHandler.calls == 1, "a capped 429 must not be retried"


def test_short_retry_after_is_retried_then_succeeds(mock_endpoint, tmp_path):
    _PlannedHandler.plan = [
        (429, {"Retry-After": "1"}, b"slow down"),
        (200, {"Content-Type": "application/json"}, json.dumps({"articles": [{"title": "ok"}]}).encode()),
    ]
    client = HttpClient(cache_dir=tmp_path, attempts=3, retry_after_cap=60.0)
    started = time.monotonic()
    payload = client.get_json(mock_endpoint)
    elapsed = time.monotonic() - started
    assert payload["articles"][0]["title"] == "ok"
    assert _PlannedHandler.calls == 2
    assert elapsed >= 1.0, "the advertised Retry-After must actually be honoured"


def test_poisoned_plaintext_cache_is_purged(mock_endpoint, tmp_path):
    _PlannedHandler.plan = [(200, {"Content-Type": "text/plain"}, b"limit notice")]
    client = HttpClient(cache_dir=tmp_path, attempts=1)
    with pytest.raises(SourceUnavailable):
        client.get_json(mock_endpoint)
    # The bad body must not have been cached: shift the plan to a JSON success and
    # the same request must hit the network again instead of reading the cache.
    _PlannedHandler.plan = [(200, {"Content-Type": "application/json"}, b'{"ok": true}')]
    _PlannedHandler.calls = 0
    assert client.get_json(mock_endpoint) == {"ok": True}
    assert _PlannedHandler.calls == 1
