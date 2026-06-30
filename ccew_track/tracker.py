"""
Object-based tracking of filtered equatorial wave systems.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

from .filter import detect_spd

# Wave tracking configuration
_WAVE_TRACK_PARAMS = {
    "Kelvin": dict(speed_thresh=30, direction="eastward", lat_bin=[-10, 10],  save_name="CCKW"),
    "TD_N":     dict(speed_thresh=20, direction="westward", lat_bin=[0,   15],  save_name="TD_N"),
    "TD_S":     dict(speed_thresh=20, direction="westward", lat_bin=[-15, 0],   save_name="TD_S"),
    "ER_N":   dict(speed_thresh=15, direction="westward", lat_bin=[0,   10],  save_name="ER_N"),
    "ER_S":   dict(speed_thresh=15, direction="westward", lat_bin=[-10, 0],   save_name="ER_S"),
    "ER_NS":  dict(speed_thresh=15, direction="westward", lat_bin=[-10, 10],  save_name="ER_NS") 
}

VALID_WAVES = list(_WAVE_TRACK_PARAMS.keys())

_LON_M = 111_000.0  # metres per degree longitude at the equator


def track(
    filtered,
    wave="Kelvin",
    normalize=None,
    init_threshold=1.0,
    cont_threshold=0.25,
    day_cut=3,
    back_allow=5,
    daily_mean=True,
):
    """
    Track wave objects in filtered precipitation data.

    Identifies local peaks in a Hovmöller (longitude–time) diagram of
    wave-filtered, normalised precipitation, then links them across time
    steps using a physical speed limit.

    Parameters
    ----------
    filtered : xr.DataArray
        Kelvin/TD/ER-filtered precipitation from ``ccew_track.filter()``.
        Must have 'time', 'lat', and 'lon' coordinates. Any temporal resolution is supported.
    wave : str
        Wave type: 'Kelvin', 'TD', 'ER_N', or 'ER_S'.
    normalize : xr.DataArray, bool, or None
        Controls normalisation of the filtered signal before peak detection.
        - None (default): divide by the std dev computed from the lat-band Hovmöller.
        - xr.DataArray: divide by this field directly (treated as std devs, not
          variance). Must share 'lat' coordinates with ``filtered``. Use this to
          pass a pre-computed climatological std dev file for consistent thresholds.
        - False: skip normalisation and track raw filtered values.
    init_threshold : float
        Minimum amplitude [std devs] required to start a new wave track.
        Default of 1.0 follows Lawton et al. (2026). 0.5 is a more permissive option to capture early wave life stages.
    cont_threshold : float
        Minimum amplitude [std devs] required to extend an existing track.
        Default of 0.25 follows Lawton et al. (2026).
    day_cut : int
        Minimum track length in days. Tracks shorter than this are discarded.
        Default is 3 days.
    back_allow : float
        Maximum degrees longitude a wave can back-track.
        Default is 5 degrees.
    daily_mean : bool
        If True (default) and the input has sub-daily time steps, collapse to
        daily means before tracking. Each daily mean spans 00–24 Z (centred
        on 12 Z). Set to False to track at the native time resolution.

    Returns
    -------
    xr.Dataset
        Dataset with two variables:
        - ``{name}_lon`` : longitude of each tracked system [degrees]
        - ``{name}_str`` : normalised amplitude of each tracked system [std devs]
        Both have dimensions (system, time). NaN where a system does not exist.

    Examples
    --------
    >>> import ccew_track
    >>> filtered = ccew_track.filter(precip, wave="Kelvin")
    >>> tracks = ccew_track.track(filtered, wave="Kelvin")
    >>> tracks["CCKW_lon"].dropna("system", how="all")
    """
    if wave not in _WAVE_TRACK_PARAMS:
        raise ValueError(f"wave must be one of {VALID_WAVES}, got '{wave}'")

    params = _WAVE_TRACK_PARAMS[wave]
    direction = params["direction"]
    lat_bin = params["lat_bin"]
    save_name = params["save_name"]
    speed_thresh = params["speed_thresh"]

    # Ensure (time, lat, lon) order
    if set(filtered.dims) >= {"time", "lat", "lon"}:
        filtered = filtered.transpose("time", "lat", "lon")

    # Detect sampling rate; optionally collapse to daily means (00-24 Z, centred on 12 Z)
    spd = detect_spd(filtered)
    if daily_mean and spd > 1:
        filtered = filtered.resample(time="1D").mean()
        spd = 1

    # Speed limit: max degrees longitude a wave can travel per timestep
    t_res = 24 / spd  # hours per timestep
    lon_limit = (t_res * 3600 * speed_thresh) / _LON_M

    # Slice to tracking latitude band
    lat_slice = filtered.sel(lat=slice(lat_bin[0], lat_bin[-1]))

    # Normalise to standard deviations
    if normalize is False:
        norm = lat_slice
    elif normalize is None:
        std_val = float(lat_slice.std())
        norm = lat_slice / (std_val if std_val > 0 else 1.0)
    else:
        std_aligned = normalize.sel(lat=slice(lat_bin[0], lat_bin[-1]))
        norm = lat_slice / std_aligned

    # Collapse to Hovmöller (time, lon)
    hov = norm.mean(dim="lat")

    # Convert lon to –180/180 and shift seam to 60 W (a quiet region)
    lon_vals = hov.lon.values.copy()
    lon_vals = (lon_vals + 180) % 360 - 180   # → –180 to +180
    hov = hov.assign_coords(lon=lon_vals).sortby("lon")
    hov = hov.rename({"lon": "longitude"})

    lon_shifted = hov.longitude.values.copy() - 240
    lon_shifted[lon_shifted < -180] += 360
    hov = hov.assign_coords(longitude=lon_shifted).sortby("longitude")

    # Run tracker
    print("Running wave tracker...")
    raw_lon, raw_val = _run_tracker(
        hov, lon_limit, back_allow,
        cont_threshold=cont_threshold,
        init_threshold=init_threshold,
        direction=direction,
    )
    print("Connecting fragmented tracks...")
    int_lon, int_val, _ = _connect_tracks(raw_lon, raw_val, lon_limit, back_allow, direction)
    print("Removing short tracks...")
    final_lon, final_val = _clean_up(int_lon, int_val, day_cut, spd)

    if final_lon is None or final_lon.ndim == 1:
        final_lon = final_lon[np.newaxis, :] if final_lon is not None else np.full((1, hov.sizes["time"]), np.nan)
        final_val = final_val[np.newaxis, :] if final_val is not None else np.full((1, hov.sizes["time"]), np.nan)

    # Shift seam back to original longitude space
    final_lon = _unshift_seam(final_lon)

    system_ids = np.arange(1, final_lon.shape[0] + 1)
    time_coord = hov.time

    lon_da = xr.DataArray(
        final_lon,
        dims=["system", "time"],
        coords={"system": system_ids, "time": time_coord},
        attrs={"description": f"Longitude of tracked {save_name}", "units": "degrees"},
    )
    str_da = xr.DataArray(
        final_val,
        dims=["system", "time"],
        coords={"system": system_ids, "time": time_coord},
        attrs={"description": f"Normalised amplitude of tracked {save_name}", "units": "std dev"},
    )

    ds = xr.merge([
        lon_da.to_dataset(name=f"{save_name}_lon"),
        str_da.to_dataset(name=f"{save_name}_str"),
    ])
    n_tracks = int((~np.isnan(final_lon).all(axis=1)).sum())
    print(f"Done. Found {n_tracks} tracks.")
    return ds


# ── private tracking functions ────────────────────────────────────────────────

def _run_tracker(data_in, lon_limit, back_allow, cont_threshold, init_threshold,
                 direction="westward", extension_size=10):
    """Core time-stepping tracker. Returns (act_lon, act_val) arrays."""
    times = pd.to_datetime(data_in.time.values)
    empty = np.ones((1, len(times))) * np.nan

    act_lon = None
    act_val = None

    for tm_i in range(len(times)):
        row = data_in.isel(time=tm_i).squeeze()
        lon_vals = row.longitude.values

        # Wrap longitude to handle seam crossings
        extended = np.concatenate((row.values[-extension_size:], row.values, row.values[:extension_size]))
        peaks_i = signal.find_peaks(extended, height=0, prominence=0)[0]
        peaks_i = [p - extension_size for p in peaks_i if extension_size <= p < len(row) + extension_size]

        lon_peak = lon_vals[peaks_i]
        val_peak = row.values[peaks_i]

        if tm_i == 0:
            act_lon = np.full((len(lon_peak), len(times)), np.nan)
            act_val = np.full((len(val_peak), len(times)), np.nan)
            for wi in range(len(lon_peak)):
                act_lon[wi, 0] = lon_peak[wi]
                act_val[wi, 0] = val_peak[wi]
            continue

        used_prev = []
        used_curr = []

        for prev_wi in range(act_lon.shape[0]):
            prev_lon = act_lon[prev_wi, tm_i - 1]
            if np.isnan(prev_lon):
                continue

            did_append = False

            # Check for seam crossing
            if direction == "eastward":
                near_seam = (180 - prev_lon) <= lon_limit
            else:
                near_seam = (prev_lon - (-180)) <= lon_limit

            if near_seam:
                if direction == "eastward":
                    possible = lon_peak >= prev_lon
                else:
                    possible = lon_peak <= prev_lon

                if not any(possible) and len(lon_peak) > 0:
                    if direction == "eastward":
                        dist_from_seam = 180 - prev_lon
                        adj_dist = (180 + np.min(lon_peak)) + dist_from_seam
                    else:
                        dist_from_seam = prev_lon - (-180)
                        adj_dist = (180 - np.max(lon_peak)) + dist_from_seam

                    if adj_dist <= lon_limit:
                        if direction == "eastward":
                            nearest_lon = np.min(lon_peak)
                        else:
                            nearest_lon = np.max(lon_peak)
                        nearest_i = np.where(lon_peak == nearest_lon)[0][0]
                        nearest_val = val_peak[nearest_i]
                        nearest_dist = abs(adj_dist)

                        if nearest_i in used_curr:
                            loc = used_curr.index(nearest_i)
                            other = used_prev[loc]
                            if not np.isnan(other):
                                other = int(other)
                                if direction == "eastward":
                                    other_seam = act_lon[other, tm_i] < 0 and act_lon[other, tm_i - 1] > 0
                                else:
                                    other_seam = act_lon[other, tm_i] > 0 and act_lon[other, tm_i - 1] < 0
                                if other_seam:
                                    if direction == "eastward":
                                        prev_dist = (180 - act_lon[other, tm_i - 1]) + (act_lon[other, tm_i] + 180)
                                    else:
                                        prev_dist = (act_lon[other, tm_i - 1] + 180) + (180 - act_lon[other, tm_i])
                                else:
                                    prev_dist = abs(act_lon[other, tm_i] - act_lon[other, tm_i - 1])
                                if nearest_dist < abs(prev_dist):
                                    act_lon[other, tm_i] = np.nan
                                    act_val[other, tm_i] = np.nan
                                    used_prev[loc] = np.nan
                                else:
                                    continue

                        if abs(nearest_val) >= cont_threshold:
                            used_prev.append(prev_wi)
                            used_curr.append(nearest_i)
                            act_lon[prev_wi, tm_i] = nearest_lon
                            act_val[prev_wi, tm_i] = nearest_val
                            did_append = True

            if not did_append:
                if direction == "eastward":
                    possible = lon_peak >= prev_lon
                else:
                    possible = lon_peak <= prev_lon

                if any(possible):
                    if direction == "eastward":
                        nearest_lon = np.array(lon_peak)[possible].min()
                    else:
                        nearest_lon = np.array(lon_peak)[possible].max()

                    nearest_i = np.where(lon_peak == nearest_lon)[0][0]
                    nearest_dist = abs(nearest_lon - prev_lon)
                    nearest_val = val_peak[nearest_i]

                    if nearest_i in used_curr:
                        # Resolve conflict: keep closer match
                        loc = used_curr.index(nearest_i)
                        other = used_prev[loc]
                        if not np.isnan(other):
                            other = int(other)
                            if direction == "eastward":
                                other_seam = act_lon[other, tm_i] < 0 and act_lon[other, tm_i - 1] > 0
                            else:
                                other_seam = act_lon[other, tm_i] > 0 and act_lon[other, tm_i - 1] < 0
                            if other_seam:
                                if direction == "eastward":
                                    prev_dist = (180 - act_lon[other, tm_i - 1]) + (act_lon[other, tm_i] + 180)
                                else:
                                    prev_dist = (act_lon[other, tm_i - 1] + 180) + (180 - act_lon[other, tm_i])
                            else:
                                prev_dist = abs(act_lon[other, tm_i] - act_lon[other, tm_i - 1])
                            if nearest_dist < abs(prev_dist):
                                act_lon[other, tm_i] = np.nan
                                act_val[other, tm_i] = np.nan
                                used_prev[loc] = np.nan
                            else:
                                continue

                    if nearest_dist <= lon_limit and abs(nearest_val) >= cont_threshold:
                        used_prev.append(prev_wi)
                        used_curr.append(nearest_i)
                        act_lon[prev_wi, tm_i] = nearest_lon
                        act_val[prev_wi, tm_i] = nearest_val

        # Initiate new tracks for unmatched peaks
        for wi in range(len(lon_peak)):
            if wi not in used_curr:
                if abs(val_peak[wi]) >= init_threshold:
                    act_lon = np.vstack([act_lon, empty])
                    act_val = np.vstack([act_val, empty])
                    act_lon[-1, tm_i] = lon_peak[wi]
                    act_val[-1, tm_i] = val_peak[wi]

    return act_lon, act_val


def _connect_tracks(act_lon, act_val, lon_limit, back_allow, direction):
    """Merge track pairs that begin where another ends."""
    lon = act_lon.copy()
    val = act_val.copy()
    st_list, end_list = [], []

    for row in range(lon.shape[0]):
        valid = np.where(~np.isnan(lon[row, :]))[0]
        st_list.append(valid[0] if len(valid) else None)
        end_list.append(valid[-1] if len(valid) else None)

    changed = []
    for i in range(len(st_list)):
        if end_list[i] is None:
            continue
        if end_list[i] in st_list:
            j = st_list.index(end_list[i])
            if i == j:
                continue
            if direction == "eastward":
                lon_diff = lon[j, end_list[i]] - lon[i, end_list[i]]
            else:
                lon_diff = lon[i, end_list[i]] - lon[j, end_list[i]]
            if abs(lon_diff) <= lon_limit and lon_diff >= -back_allow:
                avg = np.mean([lon[i, end_list[i]], lon[j, end_list[i]]])
                lon[i, end_list[i]] = avg
                lon[i, end_list[i] + 1:] = lon[j, end_list[i] + 1:]
                lon[j, end_list[i]] = avg
                lon[j, :end_list[i] + 1] = lon[i, :end_list[i] + 1]
                changed.append([i, j])

    del_rows = [c[0] for c in changed]
    lon = np.delete(lon, del_rows, axis=0)
    val = np.delete(val, del_rows, axis=0)
    return lon, val, changed


def _clean_up(act_lon, act_val, day_cut, spd=1):
    """Discard tracks shorter than day_cut days."""
    if act_lon is None:
        return None, None
    step_len = int(day_cut * spd)
    keep_lon, keep_val = [], []
    for row in range(act_lon.shape[0]):
        valid = act_lon[row, ~np.isnan(act_lon[row, :])]
        if len(valid) >= step_len:
            keep_lon.append(act_lon[row, :])
            keep_val.append(act_val[row, :])
    if not keep_lon:
        return np.full((1, act_lon.shape[1]), np.nan), np.full((1, act_val.shape[1]), np.nan)
    return np.vstack(keep_lon), np.vstack(keep_val)


def _unshift_seam(lon_arr):
    """Reverse the 60 W seam shift applied before tracking."""
    out = lon_arr.copy()
    out[lon_arr <= -60] = lon_arr[lon_arr <= -60] + 240
    out[lon_arr > -60] = lon_arr[lon_arr > -60] - 120
    return out
