"""Shared OHLC validation and repair helpers for the lakehouse price table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .lakehouse import Lakehouse


OHLC_COLUMNS = ("open", "high", "low", "close")


def valid_ohlc_mask(frame: pd.DataFrame) -> pd.Series:
    """Mark rows with finite, strictly positive values for every OHLC field."""
    if frame.empty:
        return pd.Series(index=frame.index, dtype=bool)
    if any(column not in frame.columns for column in OHLC_COLUMNS):
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame.loc[:, OHLC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.Series(np.isfinite(array).all(axis=1) & (array > 0.0).all(axis=1), index=frame.index)


def clean_ohlc_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop invalid price bars without inventing replacement OHLC values."""
    if frame.empty:
        return frame.copy(), 0
    valid = valid_ohlc_mask(frame)
    return frame.loc[valid].copy().reset_index(drop=True), int((~valid).sum())


def repair_invalid_market_prices(lake: Lakehouse) -> int:
    """Remove legacy invalid OHLC bars from an existing lakehouse and return count."""
    if not lake.exists("market_prices"):
        return 0
    prices = lake.read("market_prices")
    clean, removed = clean_ohlc_rows(prices)
    if removed:
        lake.replace("market_prices", clean)
    return removed
