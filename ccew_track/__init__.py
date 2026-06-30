"""
ccew_track: Object-based tracking of convectively coupled equatorial waves in the 1D longitudinal plane.

Typical workflow:
    import ccew_track
    filtered = ccew_track.filter(precip_da, wave='Kelvin')
    tracks   = ccew_track.track(filtered, wave='Kelvin')
"""

from .filter import filter
from .tracker import track

__all__ = ["filter", "track"]
__version__ = "0.1.0"
