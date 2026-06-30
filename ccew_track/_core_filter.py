"""
Wavenumber-frequency filtering for equatorial waves.

Adapted from Maria Gehne's tropical_diagnostics package:
    https://github.com/mgehne/tropical_diagnostics

Only the kf_filter and kf_filter_mask functions are used here,
along with the physical constants they require.
"""

import numpy as np
from scipy import signal  

pi = np.pi
re = 6.371008e6       # Earth radius [m]
g = 9.80665           # Gravitational acceleration [m s^-2]
omega = 7.292e-05     # Angular speed of Earth [rad s^-1]
beta = 2.0 * omega / re  # Beta parameter at equator


def kf_filter_mask(fftIn, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName):
    """
    Zero out FFT coefficients outside a wave's dispersion-curve region.

    Parameters
    ----------
    fftIn : ndarray
        2D complex FFT array (wavenumber x frequency).
    obsPerDay : int
        Observations per day (1 for daily, 4 for 6-hourly).
    tMin, tMax : float
        Min/max period [days] of the filtering region.
    kMin, kMax : int
        Min/max zonal wavenumber. Negative = westward.
    hMin, hMax : float
        Min/max equivalent depth [m]. Use -9999 to skip dispersion bounds.
    waveName : str
        Wave type ('Kelvin', 'TD', 'ER', 'MRG', 'IG1', 'IG2').

    Returns
    -------
    ndarray
        Filtered FFT array, same shape as fftIn.
    """
    fftData = np.copy(fftIn)
    fftData = np.transpose(fftData)
    nf, nk = fftData.shape
    fftData = fftData[:, ::-1]

    nt = (nf - 1) * 2
    jMin = int(round(nt / (tMax * obsPerDay)))
    jMax = int(round(nt / (tMin * obsPerDay)))
    jMax = np.array([jMax, nf]).min()

    if kMin < 0:
        iMin = int(round(nk + kMin))
        iMin = np.array([iMin, nk // 2]).max()
    else:
        iMin = int(round(kMin))
        iMin = np.array([iMin, nk // 2]).min()

    if kMax < 0:
        iMax = int(round(nk + kMax))
        iMax = np.array([iMax, nk // 2]).max()
    else:
        iMax = int(round(kMax))
        iMax = np.array([iMax, nk // 2]).min()

    if jMin > 0:
        fftData[0:jMin, :] = 0
    if jMax < nf:
        fftData[jMax + 1:nf, :] = 0
    if iMin < iMax:
        if iMin > 0:
            fftData[:, 0:iMin] = 0
        if iMax < nk:
            fftData[:, iMax + 1:nk] = 0
    else:
        fftData[:, iMax + 1:iMin] = 0

    c = np.empty([2])
    if hMin == -9999:
        c[0] = np.nan
        c[1] = np.nan if hMax == -9999 else np.nan
    else:
        c = np.sqrt(g * np.array([hMin, hMax])) if hMax != -9999 else np.array([np.sqrt(g * hMin), np.nan])

    spc = 24 * 3600.0 / (2 * pi * obsPerDay)  # seconds per cycle

    for i in range(nk):
        if i < (nk / 2):
            k = i / re
        else:
            k = -(nk - i) / re

        jMinWave = 0
        jMaxWave = nf

        wn = waveName.upper()
        if wn in ("KELVIN",):
            ftmp = k * c
            freq = np.array(ftmp)
        elif wn in ("ER",):
            ftmp = -beta * k / (k ** 2 + 3 * beta / c)
            freq = np.array(ftmp)
        elif wn in ("MRG", "IG0"):
            if k == 0:
                ftmp = np.sqrt(beta * c)
                freq = np.array(ftmp)
            elif k > 0:
                ftmp = k * c * (0.5 + 0.5 * np.sqrt(1 + 4 * beta / (k ** 2 * c)))
                freq = np.array(ftmp)
            else:
                ftmp = k * c * (0.5 - 0.5 * np.sqrt(1 + 4 * beta / (k ** 2 * c)))
                freq = np.array(ftmp)
        elif wn in ("IG1",):
            ftmp = np.sqrt(3 * beta * c + k ** 2 * c ** 2)
            freq = np.array(ftmp)
        elif wn in ("IG2",):
            ftmp = np.sqrt(5 * beta * c + k ** 2 * c ** 2)
            freq = np.array(ftmp)
        else:
            freq = np.array([0, nf]) / spc  # TD-band: no dispersion bounds

        if hMin == -9999:
            jMinWave = 0
        else:
            jMinWave = int(np.floor(freq[0] * spc * nt))
        if hMax == -9999:
            jMaxWave = nf
        else:
            jMaxWave = int(np.ceil(freq[1] * spc * nt))

        jMaxWave = np.array([jMaxWave, 0]).max()
        jMinWave = np.array([jMinWave, nf]).min()

        if jMinWave > 0:
            fftData[0:jMinWave, i] = 0
        if jMaxWave < nf:
            fftData[jMaxWave + 1:nf, i] = 0

    fftData = fftData[:, ::-1]
    fftData = np.transpose(fftData)
    return fftData


def kf_filter(data, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName):
    """
    Filter a 2D (time x lon) array in wavenumber-frequency space.

    Parameters
    ----------
    data : ndarray, shape (time, lon)
        Input 2D array for a single latitude.
    obsPerDay : int
        Observations per day.
    tMin, tMax, kMin, kMax, hMin, hMax : float
        Filter region parameters (see kf_filter_mask).
    waveName : str
        Wave type name.

    Returns
    -------
    ndarray, shape (time, lon)
        Filtered data.
    """
    data = np.transpose(data, axes=[1, 0])          # → (lon, time)
    fftdata = np.fft.rfft2(data, axes=(0, 1))
    fftfilt = kf_filter_mask(fftdata, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName)
    datafilt = np.fft.irfft2(fftfilt, axes=(0, 1))
    datafilt = np.transpose(datafilt, axes=[1, 0])  # → (time, lon)
    return datafilt
