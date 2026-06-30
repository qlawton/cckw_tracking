"""
ccew_track: Object-based tracking of convectively coupled equatorial waves in the 1D longitudinal plane.

Typical workflow:
    import ccew_track
    filtered = ccew_track.filter(precip_da, wave='Kelvin')
    tracks   = ccew_track.track(filtered, wave='Kelvin')
"""

from .filter import filter
from .tracker import track

__all__ = ["filter", "track", "fetch_example_data"]
__version__ = "0.1.0"

_EXAMPLE_URL  = "https://zenodo.org/records/21054811/files/imerg_year_2018.nc"
_EXAMPLE_HASH = "md5:40511a3a4000102b0d79e762038582e2"


def fetch_example_data(data_dir=None):
    """
    Download the example IMERG 2018 dataset from Zenodo.

    The file is ~60 MB and is cached after the first download — subsequent
    calls return immediately. Requires the optional ``pooch`` dependency:
    ``pip install pooch``.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory to save the file. Defaults to ``example_data/`` inside
        the repository root (one level above the package).

    Returns
    -------
    str
        Local path to ``imerg_year_2018.nc``.

    Examples
    --------
    >>> import xarray as xr, ccew_track
    >>> path = ccew_track.fetch_example_data()
    >>> precip = xr.open_dataset(path).precipitation
    """
    try:
        import pooch
    except ImportError:
        raise ImportError(
            "pooch is required to download example data. "
            "Install it with:  pip install pooch"
        )

    import pathlib
    if data_dir is None:
        data_dir = pathlib.Path(__file__).parent.parent / "example_data"

    return pooch.retrieve(
        url=_EXAMPLE_URL,
        known_hash=_EXAMPLE_HASH,
        fname="imerg_year_2018.nc",
        path=str(data_dir),
        progressbar=False,
    )
