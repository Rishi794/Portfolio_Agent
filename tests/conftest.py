"""Shared fixtures: configuration, a deterministic synthetic lakehouse and one cached
end-to-end pipeline run (the whole CI suite makes zero network calls)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import Config
from src.storage.lakehouse import Lakehouse
from src.synthetic import generate_synthetic_lakehouse

SYNTHETIC_DAYS = 260
SYNTHETIC_TICKERS = 48


@pytest.fixture(scope="session", autouse=True)
def _hermetic_forecaster() -> None:
    """Force the Timestral Stage-3 engine into its offline fallback for the whole run.

    The CI suite makes zero network calls; without this, any job that happens to have
    the ``timestral`` extra installed (and the fan-in's 837 MB HF cache) would pull the
    checkpoint into unit tests.
    """
    os.environ["PMS_TIMESTRAL_DISABLED"] = "1"


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.load()


@pytest.fixture()
def lake(tmp_path: Path) -> Lakehouse:
    return Lakehouse(tmp_path / "lake")


@pytest.fixture(scope="session")
def synthetic_shards(tmp_path_factory: pytest.TempPathFactory, config: Config) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    return generate_synthetic_lakehouse(
        root / "data_lake",
        config,
        days=SYNTHETIC_DAYS,
        n_tickers_per_market=SYNTHETIC_TICKERS,
    )


@pytest.fixture(scope="session")
def e2e(tmp_path_factory: pytest.TempPathFactory, config: Config, synthetic_shards: Path) -> dict:
    """Run the reducer once (Stage-3 Fan-In) and hand the artifacts to the tests."""
    from src.main_v4 import V4AutonomousPortfolioReducer

    root = synthetic_shards.parent
    lake_dir = root / "data_lake"
    state_dir = root / "state"
    reducer = V4AutonomousPortfolioReducer(
        config,
        lake_dir,
        state_dir,
        shards_dir=synthetic_shards,
        embed_backend="hashing",
    )
    summary = reducer.run(["US", "INDIA"], mode="synthetic_demo")
    return {
        "reducer": reducer,
        "summary": summary,
        "lake_dir": lake_dir,
        "state_dir": state_dir,
        "shards": synthetic_shards,
    }
