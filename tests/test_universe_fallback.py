"""Universe source-chain tests.

SEC rows a 403 for generic User-Agents (and sometimes for datacenter IPs), so a
blocked primary source must degrade to the configured fallback instead of
killing the whole shard job. These tests use a fake client: zero network.
"""

from __future__ import annotations

from typing import Any

from src.config import Config
from src.ingestion.base import SourceUnavailable, compliant_user_agent
from src.ingestion.universe import UniverseLoader

NASDAQ_LISTED = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n"
    "TESTQ|Test Corp - Common Stock|Q|Y|N|100|N|N\n"
    "File Creation Time: 1001202518:00|||||||\n"
)

OTHER_LISTED = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "BRK.A|Berkshire Hathaway Inc. Class A Common Stock|N|BRK.A|N|1|N|BRK.A\n"
    "XYZ.PA|XYZ Corp 6.50% Preferred Series A|N|XYZ.PA|N|100|N|XYZ.PA\n"
)


class _FallbackClient:
    """SEC answers 403; the NASDAQ symbol directory answers normally."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def get_json(self, url: str, params: dict[str, Any] | None = None, extra_headers: dict[str, str] | None = None) -> Any:
        raise SourceUnavailable(f"GET {url} failed: Client error '403 Forbidden'")

    def get_text(self, url: str, params: dict[str, Any] | None = None, extra_headers: dict[str, str] | None = None) -> str:
        return self.texts[url]


def _loader() -> UniverseLoader:
    texts = {
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt": NASDAQ_LISTED,
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt": OTHER_LISTED,
    }
    return UniverseLoader(Config.load(), _FallbackClient(texts))


def test_us_universe_falls_back_to_symbol_directory() -> None:
    frame = _loader().load_us()
    tickers = set(frame["ticker"])
    assert {"AAPL", "MSFT", "BRK-A"}.issubset(tickers)  # class shares normalised to SEC/Yahoo style
    assert "TESTQ" not in tickers  # test issues filtered
    assert "XYZ-PA" not in tickers  # preferreds filtered
    assert frame["market"].unique().tolist() == ["US"]
    assert frame["ticker"].is_monotonic_increasing


def test_us_universe_raises_when_every_source_fails() -> None:
    class _DeadClient:
        def get_json(self, url: str, params: dict[str, Any] | None = None, extra_headers: dict[str, str] | None = None) -> Any:
            raise SourceUnavailable("403")

        def get_text(self, url: str, params: dict[str, Any] | None = None, extra_headers: dict[str, str] | None = None) -> str:
            raise SourceUnavailable("403")

    loader = UniverseLoader(Config.load(), _DeadClient())
    try:
        loader.load_us()
    except SourceUnavailable as exc:
        assert "sec_company_tickers" in str(exc) and "nasdaq_symdir" in str(exc)
    else:  # pragma: no cover - the chain must not silently succeed
        raise AssertionError("expected SourceUnavailable when every configured source fails")


def test_sec_user_agent_carries_a_contact(monkeypatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert "@" in compliant_user_agent()  # SEC requires a declared contact
    monkeypatch.setenv("SEC_USER_AGENT", "Custom/1.0 (ops@example.com)")
    assert compliant_user_agent() == "Custom/1.0 (ops@example.com)"
