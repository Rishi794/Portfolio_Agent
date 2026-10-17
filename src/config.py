"""Configuration loading.

All tunable numbers live in ``config/*.yaml`` (see ``config/engine.yaml`` for the
documented priors). Code never embeds a statistical constant: it reads it from
here so that a run is fully reproducible from ``(config, data, run_id)``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Read-only, dotted-path view over the YAML configuration tree."""

    def __init__(self, raw: dict[str, Any], root: Path) -> None:
        self._raw = raw
        self.root = root

    # ------------------------------------------------------------------ load --
    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> "Config":
        root = Path(
            config_dir or os.getenv("PMS_CONFIG_DIR") or (REPO_ROOT / "config")
        ).expanduser()
        raw: dict[str, Any] = {}
        for path in sorted(list(root.glob("*.yaml")) + list(root.glob("*.yml"))):
            with path.open("r", encoding="utf-8") as fh:
                raw[path.stem] = yaml.safe_load(fh) or {}
        return cls(raw, root)

    # --------------------------------------------------------------- accessors --
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {dotted}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        value = self._raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"config section {name!r} is not a mapping")
        return value

    # ------------------------------------------------------------------ domain --
    @property
    def engine(self) -> dict[str, Any]:
        return self.get("engine", {}) or {}

    def e(self, dotted: str, default: Any = None) -> Any:
        """Shorthand for ``engine.<dotted>``."""
        return self.get(f"engine.{dotted}", default)

    @property
    def pillar_order(self) -> list[str]:
        return list(self.e("pillar_order", []) or self.get("regime_weights.pillar_order", []))

    def regime_weights(self) -> dict[str, np.ndarray]:
        table = self.get("regime_weights.regimes", {}) or {}
        out: dict[str, np.ndarray] = {}
        for regime, spec in table.items():
            weights = np.asarray(spec["weights"], dtype=float)
            out[regime] = weights / weights.sum()
        return out

    def regime_cash_range(self, regime: str) -> tuple[float, float]:
        spec = self.get(f"regime_weights.regimes.{regime}", {}) or {}
        low, high = spec.get("cash_range", [self.e("turbulence.cash_min", 0.03), self.e("turbulence.cash_max", 0.45)])
        return float(low), float(high)

    def fallback_regime(self) -> str:
        return str(self.get("regime_weights.fallback_regime", "LOW_VOL_EXPANSION"))

    def markets(self) -> list[str]:
        return list((self.get("universe.markets", {}) or {}).keys())

    def market(self, name: str) -> dict[str, Any]:
        markets = self.get("universe.markets", {}) or {}
        if name not in markets:
            raise KeyError(f"unknown market {name!r}; configured: {sorted(markets)}")
        return dict(markets[name])

    def cost_profile(self, market: str) -> dict[str, float]:
        profile = self.market(market).get("statutory_cost_profile", market)
        return dict(self.e(f"costs.markets.{profile}", {}) or {})

    def half_life_default(self, domain: str) -> float:
        table = self.e("events.half_life_default_days", {}) or {}
        if domain in table:
            return float(table[domain])
        for key, value in table.items():
            if domain.startswith(key):
                return float(value)
        return float(self.e("events.half_life_default_days.GEOPOLITICAL", 14.0))

    # --------------------------------------------------------------- integrity --
    def fingerprint(self) -> str:
        payload = json.dumps(self._raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@functools.lru_cache(maxsize=4)
def get_config(config_dir: str | None = None) -> Config:
    return Config.load(config_dir)
