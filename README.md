# ccew_track

Object-based tracking of convectively coupled equatorial waves (CCEWs) in the longitudinal plane.

`ccew_track` filters precipitation data in wavenumber-frequency space and tracks the resulting wave objects forward in time using a physical speed limit. It was primarily built for Convectively Coupled Kelvin Waves (CCKWs), but also supports the tropical depression (TD) band and equatorial Rossby waves (n=1). Only CCKW results have been fully tested and published.

Written by **Quinton A. Lawton** (Northern Illinois University; formerly NSF National Center for Atmospheric Research).

---

## Citation

If you use this package, please cite the repository and the paper describing the methodology:

> Lawton, Q. A., R. Rios-Berrios, F. Judt, L. Magnusson, and M. Köhler, 2026: On the Representation of Convectively Coupled Kelvin Waves in Operational Forecast Models: An Object-Tracking Perspective. *Weather and Forecasting*, 41(5), 1073–1090. https://doi.org/10.1175/WAF-D-25-0182.1

The wavenumber-frequency filtering follows [Wheeler and Kiladis (1999)](https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2). The `kf_filter` function is adapted from Maria Gehne's [tropical_diagnostics](https://github.com/mgehne/tropical_diagnostics) package.

---

## Installation

```bash
# From the repo root
pip install -e .
```

Dependencies (`numpy`, `xarray`, `scipy`, `pandas`) are installed automatically. To include `matplotlib` for plotting:

```bash
pip install -e ".[plot]"
```

A fresh conda environment with everything needed:

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

precip = xr.open_dataset("example_data/imerg_year_2018.nc").precipitation

# Step 1: filter to the Kelvin wave band
filtered = ccew_track.filter(precip, wave="Kelvin")

# Step 2: track wave objects forward in time
tracks = ccew_track.track(filtered, wave="Kelvin")

# tracks is an xr.Dataset with dimensions (system, time)
# CCKW_lon: longitude of each tracked wave [degrees]
# CCKW_str: normalised amplitude [std devs]
print(tracks)
```

See `notebooks/example_tracking.ipynb` for a full walkthrough including a Hovmöller diagram with overlaid tracks.

---

## API

### `ccew_track.filter(precip, wave="Kelvin", lat_min=-10, lat_max=20, pad_days=30)`

Applies a wavenumber-frequency bandpass filter following [Wheeler and Kiladis (1999)](https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2).

| Parameter | Description |
|-----------|-------------|
| `precip` | `xr.DataArray` with `time`, `lat`, `lon` coordinates. Any temporal resolution. |
| `wave` | `"Kelvin"`, `"TD"`, `"ER"`, `"MRG"`, or `"IG1"` |
| `lat_min`, `lat_max` | Latitude band for filtering (default −10 to 20 °N) |
| `pad_days` | Days of zero-padding at each end to reduce edge effects (default 30) |

Returns an `xr.DataArray` of filtered precipitation on the same grid, sliced to `[lat_min, lat_max]`.

**Filter windows:**

| Wave | Period (days) | Wavenumber | Equiv. depth (m) |
|------|--------------|------------|-----------------|
| Kelvin | 2.5 – 20 | 1 – 14 | 8 – 90 |
| TD | 2 – 8 | −27 – −6 | — |
| ER | 10 – 40 | −10 – −1 | 8 – 90 |
| MRG | 2 – 8 | −10 – −1 | 8 – 90 |
| IG1 | 1.2 – 2.6 | −15 – −1 | 12 – 90 |

---

### `ccew_track.track(filtered, wave="Kelvin", normalize=None, init_threshold=1.0, cont_threshold=0.25, day_cut=3, back_allow=5, daily_mean=True)`

Finds local peaks in a Hovmöller diagram of the filtered signal and links them forward (or backward for westward waves) in time using a physical speed limit. See [Lawton et al. (2026)](https://doi.org/10.1175/WAF-D-25-0182.1) for a full description of the algorithm.

| Parameter | Description |
|-----------|-------------|
| `filtered` | Output of `ccew_track.filter()` |
| `wave` | `"Kelvin"`, `"TD_N"` (0–15 °N), `"TD_S"` (15 °S–0), `"ER_N"` (0–10 °N), `"ER_S"` (10 °S–0), `"ER_NS"` (10 °S–10 °N) |
| `normalize` | `None`: normalise by the std dev of the Hovmöller (default). `xr.DataArray`: divide by this field directly — must be in std dev units, not variance. `False`: skip normalisation and track raw values. |
| `init_threshold` | Minimum amplitude (std devs) to start a new track. Default 1.0. |
| `cont_threshold` | Minimum amplitude (std devs) to extend an existing track. Default 0.25. |
| `day_cut` | Minimum track length in days. Default 3. |
| `back_allow` | Degrees of back-tracking allowed when connecting tracks. Default 5°. |
| `daily_mean` | If `True` (default) and input is sub-daily, collapse to daily means (00–24 Z) before tracking. |

Returns an `xr.Dataset` with variables `{name}_lon` and `{name}_str`, both dimensioned `(system, time)`.

**Speed limits by wave type:**

| Wave | Max speed | Direction |
|------|-----------|-----------|
| Kelvin | 30 m/s | Eastward |
| TD | 20 m/s | Westward |
| ER | 15 m/s | Westward |

---

## Example data

The notebook uses `example_data/imerg_year_2018.nc` — a year of 6-hourly [IMERG Late Run v7](https://doi.org/10.5067/GPM/IMERG/3B-HH-L/07) precipitation remapped to 1° × 1° over the tropical band (10 °S–20 °N). The file is hosted on Zenodo (doi:[10.5281/zenodo.21054811](https://doi.org/10.5281/zenodo.21054811)) and can be downloaded directly from Python:

```python
# requires: pip install pooch
import ccew_track
path = ccew_track.fetch_example_data()   # ~60 MB, cached after first download
```

Or manually from the [Zenodo record](https://zenodo.org/records/21054811). Place the file in `example_data/` before running the notebook.

---

## Folder structure

```
ccew_track/          — installable Python package
notebooks/           — example Jupyter notebooks
example_data/        — place downloaded example data here
forecast_metrics/    — standalone scripts for DIMOSIC forecast skill metrics
legacy/              — original flat scripts, kept for reference
```

---

## Acknowledgements

This material is based upon work supported by the NSF National Center for Atmospheric Research, which is a major facility sponsored by the U.S. National Science Foundation under Cooperative Agreement No. 1852977.

The `kf_filter` and `kf_filter_mask` functions in `ccew_track/_core_filter.py` are adapted from the [tropical_diagnostics](https://github.com/mgehne/tropical_diagnostics) package by Maria Gehne (NOAA PSL).
