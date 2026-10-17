"""Filtering / signal-processing layer."""

from .embedding_projector import EventProjector, get_backend
from .kalman_smoother import AdaptiveKalmanFilter
from .lowdin_orthogonalizer import lowdin_orthogonalize
from .regime_turbulence import MacroRegimeDetector, RegimeAllocation
from .self_calibrating_engine import SelfCalibratingQuantEngine

__all__ = [
    "AdaptiveKalmanFilter",
    "EventProjector",
    "MacroRegimeDetector",
    "RegimeAllocation",
    "SelfCalibratingQuantEngine",
    "get_backend",
    "lowdin_orthogonalize",
]
