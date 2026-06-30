"""
Wavenumber-frequency filtering for equatorial wave types.
"""

import numpy as np
import pandas as pd
import xarray as xr

from ._core_filter import kf_filter

# Filter parameters from Wheeler & Kiladis (1999) and Russell & Aiyyer (2020)
_WAVE_PARAMS = {
    "Kelvin": dict(tMin=2.5, tMax=20,  kMin=1,   kMax=14,  hMin=8,     hMax=90),
    "TD":     dict(tMin=2,   tMax=8,   kMin=-27,  kMax=-6,  hMin=-9999, hMax=-9999),
    "ER":     dict(tMin=10,  tMax=40,  kMin=-10,  kMax=-1,  hMin=8,     hMax=90),
    "MRG":    dict(tMin=2,   tMax=8,   kMin=-10,  kMax=-1,  hMin=8,     hMax=90),
    "IG1":    dict(tMin=1.2, tMax=2.6, kMin=-15,  kMax=-1,  hMin=12,    hMax=90),
}

VALID_WAVES = list(_WAVE_PARAMS.keys())


def filter(precip, wave="Kelvin", lat_min=-10, lat_max=20, pad_days=30):
    """
    Filter precipitation data for a specified equatorial wave type.

    Applies a wavenumber-frequency bandpass filter following the method of
    Wheeler & Kiladis (1999). Zero-padding is added at both ends to reduce
    edge effects and then removed before returning.

    Parameters
    ----------
    precip : xr.DataArray
        Precipitation data with 'time', 'lat', and 'lon' coordinates.
        Dimension order does not matter. Units are arbitrary (mm/hr, mm/day, etc.).
    wave : str
        Wave type to filter for. One of: 'Kelvin', 'TD', 'ER', 'MRG', 'IG1'.
    lat_min, lat_max : float
        Latitude range [degrees] over which to apply the filter.
    pad_days : int
        Days of zero-padding added at each end before filtering. Padding is
        removed from the output. Default of 30 days is sufficient for most
        wave types.

    Returns
    -------
    xr.DataArray
        Filtered precipitation with the same time/lat/lon coordinates as the
        (possibly daily-averaged) input, sliced to [lat_min, lat_max].

    Raises
    ------
    ValueError
        If ``wave`` is not a recognised wave type.

    Examples
    --------
    >>> import xarray as xr
    >>> import ccew_track
    >>> precip = xr.open_dataset("imerg_year_2018.nc").precipitation
    >>> filtered = ccew_track.filter(precip, wave="Kelvin")
    """
    if wave not in _WAVE_PARAMS:
        raise ValueError(f"wave must be one of {VALID_WAVES}, got '{wave}'")

    # Ensure dimensions are (time, lat, lon)
    precip = standardise_dims(precip)

    # Slice to requested latitude band
    precip = precip.sel(lat=slice(lat_min, lat_max))

    # Replace fill values with 0
    precip = precip.fillna(0.0)

    # Optionally compute daily means
    spd = detect_spd(precip)

    pad_steps = pad_days * spd
    params = _WAVE_PARAMS[wave]

    lats = precip.lat.values
    n_time = precip.sizes["time"]
    n_lon = precip.sizes["lon"]
    filtered_arr = np.zeros((n_time, len(lats), n_lon))

    print(f"Filtering {wave} wave across {len(lats)} latitudes...")
    for li, _ in enumerate(lats):
        row = precip.isel(lat=li).values  # (time, lon)
        row_padded = np.pad(row, ((pad_steps, pad_steps), (0, 0)), mode="constant")
        row_filt = kf_filter(row_padded, spd, **params, waveName=wave)
        filtered_arr[:, li, :] = row_filt[pad_steps: pad_steps + n_time, :]

    result = xr.DataArray(
        filtered_arr,
        coords={"time": precip.time, "lat": precip.lat, "lon": precip.lon},
        dims=["time", "lat", "lon"],
        attrs={"wave": wave, "description": f"{wave}-filtered precipitation"},
    )
    return result


# ── helpers ──────────────────────────────────────────────────────────────────

def standardise_dims(da):
    """Transpose DataArray to (time, lat, lon) regardless of input order."""
    required = {"time", "lat", "lon"}
    missing = required - set(da.dims)
    if missing:
        raise ValueError(f"precip is missing dimensions: {missing}")
    return da.transpose("time", "lat", "lon")


def detect_spd(da):
    """Infer observations-per-day from the time coordinate."""
    times = pd.to_datetime(da.time.values)
    if len(times) < 2:
        return 1
    dt_hours = (times[1] - times[0]).total_seconds() / 3600
    return max(1, int(round(24 / dt_hours)))
