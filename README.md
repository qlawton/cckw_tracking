# ccew_track

**Object-based tracking of convectively coupled equatorial waves (CCEWs) in the longitudinal plane.**

`ccew_track` is a Python package that filters precipitation data in wavenumber-frequency space and tracks the resulting wave objects forward in time using a physical speed limit. It is primarily designed for Convectively Coupled Kelvin Waves (CCKWs) but also supports the tropical depression (TD) band and equatorial Rossby waves (n=1). Only CCKW results have been fully tested and published.

Written by **Quinton A. Lawton** (Northern Illinois University; formerly NSF National Center for Atmospheric Research). Partially supported by NSF NCAR under Cooperative Agreement 1852977.

---

## Citation

If you use this package in your research, please cite both the repository and the paper that describes the methodology:

> Lawton, Q. A., R. Rios-Berrios, F. Judt, L. Magnusson, and M. Köhler, 2025: On the Representation of Convectively Coupled Kelvin Waves in Operational Forecast Models: An Object-Tracking Perspective. *Weather and Forecasting*, **41**(5), 1073–1090. https://doi.org/10.1175/WAF-D-25-0182.1

The wavenumber-frequency filtering follows:

> Wheeler, M., and G. N. Kiladis, 1999: Convectively Coupled Equatorial Waves: Analysis of Clouds and Temperature in the Wavenumber–Frequency Domain. *J. Atmos. Sci.*, **56**, 374–399.

The `kf_filter` function is adapted from Maria Gehne's [tropical_diagnostics](https://github.com/mgehne/tropical_diagnostics) package.

---

## Installation

```bash
# From the repo root — editable install into your active environment
pip install -e .
```

Required dependencies (installed automatically): `numpy`, `xarray`, `scipy`, `pandas`.  
Optional plotting dependency: `pip install -e ".[plot]"` (adds `matplotlib`).

A conda environment with all dependencies can be created with:

```bash
conda create -n ccew-track -c conda-forge python=3.11 numpy xarray scipy pandas matplotlib notebook
conda activate ccew-track
pip install -e .
```

---

## Quick start

```python
import xarray as xr
import ccew_track

# Load precipitation data (xr.DataArray with time, lat, lon coordinates)
precip = xr.open_dataset("example_data/imerg_year_2018.nc").precipitation

# Step 1 — wavenumber-frequency filter
filtered = ccew_track.filter(precip, wave="Kelvin")

# Step 2 — object-based tracking
tracks = ccew_track.track(filtered, wave="Kelvin")

# tracks is an xr.Dataset with dimensions (system, time)
print(tracks)
# Data variables:
#   CCKW_lon  (system, time)  -- longitude of each tracked wave [degrees]
#   CCKW_str  (system, time)  -- normalised amplitude [std devs]
```

See `notebooks/example_tracking.ipynb` for a worked example including a Hovmöller diagram with overlaid tracks.

---

## API

### `ccew_track.filter(precip, wave="Kelvin", lat_min=-10, lat_max=20, pad_days=30)`

Applies a Wheeler & Kiladis (1999) wavenumber-frequency bandpass filter to precipitation data.

| Parameter | Description |
|-----------|-------------|
| `precip` | `xr.DataArray` with `time`, `lat`, `lon` coordinates. Any temporal resolution. |
| `wave` | Wave type: `"Kelvin"`, `"TD"`, `"ER"`, `"MRG"`, or `"IG1"`. |
| `lat_min`, `lat_max` | Latitude band for filtering (default −10 to 20 °N). |
| `pad_days` | Days of zero-padding at each end to reduce edge effects (default 30). |

Returns an `xr.DataArray` of filtered precipitation on the same grid, sliced to `[lat_min, lat_max]`.

**Filter windows** (Wheeler & Kiladis 1999; Russell & Aiyyer 2020):

| Wave | Period (days) | Wavenumber | Equiv. depth (m) |
|------|--------------|------------|-----------------|
| Kelvin | 2.5 – 20 | 1 – 14 | 8 – 90 |
| TD | 2 – 8 | −27 – −6 | — |
| ER | 10 – 40 | −10 – −1 | 8 – 90 |
| MRG | 2 – 8 | −10 – −1 | 8 – 90 |
| IG1 | 1.2 – 2.6 | −15 – −1 | 12 – 90 |

---

### `ccew_track.track(filtered, wave="Kelvin", normalize=None, init_threshold=1.0, cont_threshold=0.25, day_cut=3, back_allow=5, daily_mean=True)`

Identifies local peaks in a Hovmöller diagram of the filtered signal and links them across time steps using a physical speed limit.

| Parameter | Description |
|-----------|-------------|
| `filtered` | Output of `ccew_track.filter()`. |
| `wave` | `"Kelvin"`, `"TD_N"` (0–15 °N), `"TD_S"` (15 °S–0), `"ER_N"` (0–10 °N), `"ER_S"` (10 °S–0), `"ER_NS"` (10 °S–10 °N). |
| `normalize` | `None` (default): normalise by std dev of the Hovmöller. `xr.DataArray`: divide by this field directly (must be std devs, not variance). `False`: track raw filtered values. |
| `init_threshold` | Minimum amplitude (std devs) to start a new track. Default 1.0. |
| `cont_threshold` | Minimum amplitude (std devs) to extend an existing track. Default 0.25. |
| `day_cut` | Minimum track length in days. Default 3. |
| `back_allow` | Degrees of back-tracking allowed when connecting tracks. Default 5°. |
| `daily_mean` | If `True` (default) and input is sub-daily, collapse to daily means (00–24 Z) before tracking. |

Returns an `xr.Dataset` with variables `{wave}_lon` and `{wave}_str`, both with dimensions `(system, time)`.

**Tracking speed limits:**

| Wave type | Max speed | Direction |
|-----------|-----------|-----------|
| Kelvin | 30 m/s | Eastward |
| TD | 20 m/s | Westward |
| ER | 15 m/s | Westward |

---

## Folder structure

```
ccew_track/          # Installable Python package
  __init__.py
  filter.py          # ccew_track.filter()
  tracker.py         # ccew_track.track()
  _core_filter.py    # Vendored kf_filter from tropical_diagnostics

notebooks/           # Example Jupyter notebooks
  example_tracking.ipynb

example_data/        # Placeholder — see below for data

forecast_metrics/    # Standalone scripts for DIMOSIC forecast skill metrics
                     # (Lawton et al. 2026)

legacy/              # Original flat scripts, preserved for reference
                     # Superseded by the ccew_track package
```

### Example data

`example_data/imerg_year_2018.nc` (60 MB) is not tracked in this repository due to its size. It contains 6-hourly IMERG Late Run precipitation for 2018, sliced to the tropical band (10 °S–20 °N) at 1 × 1° resolution. To reproduce the notebook, place this file in `example_data/`.

---

## Acknowledgements

The `kf_filter` and `kf_filter_mask` functions in `ccew_track/_core_filter.py` are adapted from the [tropical_diagnostics](https://github.com/mgehne/tropical_diagnostics) package by Maria Gehne (NOAA PSL). See that repository for the full spectral analysis toolkit.
