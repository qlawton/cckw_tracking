#!/usr/bin/env python
# coding: utf-8

"""
FFT_filter_imerg_daily_mean.py

Adapted from Maria Gehne's "Tropical Diagnostics" package  
Author: Q. Lawton, NCAR  
Date: 2024-01-15

Description:
    This script reads precipitation data from a user-specified NetCDF file and applies 
    filtering in wavenumber-frequency space for a specified region. 
    The user must set:
        - Input and output file paths
        - Start and end time of the data
        - Filtering region parameters
        - Latitude range to filter
        - Number of observations per day

Note:
    Filtering can be slow for high spatial and temporal resolution datasets.
"""
import numpy as np
import xarray as xr
import pandas as pd
from tropical_diagnostics import spacetime as st
import time as systime
import sys

# file and pathname
filein = 'combined_1998_2024_daily_mean_IMERG_with_late_run.nc'
pathin = '/glade/work/qlawton/DATA/IMERG/'
pathout = '/glade/work/qlawton/DATA/IMERG/DAILY_FILTERED_WITH_LATE/'

daily = True
input_6hr = False
# If input_6hr is True, then the data is 6-hourly and daily is False.
# If input_6hr is False, then the data is daily and daily is True.
# If daily is True, then the data is daily and input_6hr is False.

### List of waves to loop over
wavelist = ['TD'] #Can be 'Kelvin', 'MRG', 'IG1', 'ER', 'TD' 

# number of obs per day
if daily==True:
    spd = 1
else:
    spd = 4
pad_num=int(30*spd) # # number of time steps to pad the data with zeros at the beginning and end

datestrt = "1998-01-01" # start date of the data
datelast = "2024-10-30-18" # end date of the data

for wvi in range(len(wavelist)):
    waveName = wavelist[wvi]

    # filename for filtered data
    if daily==True:
        fileout = 'daily_mean_padded_imerg-daily'+'.'+waveName
    else:
        fileout = 'daily_mean_padded_imerg-6hr'+'.'+waveName
    print(fileout)   
    # parameters for filtering the data
    latMin = -10
    latMax = 20
    

    # values for filtering regions
    if waveName == "Kelvin":
        # Filter for Kelvin band
        tMin = 2.5
        tMax = 20
        kMin = 1
        kMax = 14
        hMin = 8
        hMax = 90
    elif waveName == "MRG":
        # Filter for 2-6 day bandpass
        # Updated to WH definition (f
        tMin = 2
        tMax = 8
        kMin = -10
        kMax = -1
        hMin = 8
        hMax = 90
        #hMin = -9999
        #hMax = -9999
    elif waveName == "IG1":
        # Filter for WIGs
        tMin = 1.2
        tMax = 2.6
        kMin = -15
        kMax = -1
        hMin = 12
        hMax = 90
    elif waveName == "ER":
        tMin = 10
        tMax = 40
        kMin = -10
        kMax = -1
        hMin = 8
        hMax = 90
    elif waveName == "TD":
        # Matching the Russell and Aiyyer (2020) method
        tMin = 2 
        tMax = 8
        kMin = -27 #Around 1500km #Super large to keep all small wavenumbers 
        kMax = -6 #Around 6500km
        hMin = -9999
        hMax = -9999
    
    
    
    # read data from file
    print("open data set...")
    ds = xr.open_dataset(pathin+filein).sel(time=slice(datestrt, datelast))#.drop_vars("time_bnds")
    ds = ds.transpose('time', 'lat', 'lon')
    # Check for and fill missing values
    actual_times = pd.to_datetime(ds['time'].values)

    #expected_times = pd.date_range(start=actual_times[0], end=actual_times[-1], freq='6H')

    #print('Reindexing...')
    #ds = ds.reindex(time=expected_times)
    if daily == True and input_6hr == True:
        # Select the third timestep of each day (e.g., 12 UTC) after grouping by date.
        # This is to center the daily means on 12Z, which aligns with the center of NOAA OLR daily means for comparision.
        ds = ds.groupby('time.date').apply(lambda x: x.isel(time=2)).swap_dims({'date':'time'}).drop_vars('date')
    data = ds['precipitation'].sel(lat=slice(latMin, latMax),time=slice(datestrt, datelast)).pad(time=pad_num, constant_values = 0).fillna(0).load()
    lat = ds['lat'].sel(lat=slice(latMin, latMax))
    lon = ds['lon']
    time = ds['time'].sel(time=slice(datestrt, datelast)).pad(time=pad_num, constant_values = 0)
    ds.close()
    print("done. size of data array:")
    print(data.shape)
    
    # filter each latitude
    datafilt = xr.DataArray(np.zeros(data.shape), dims=['time', 'lat', 'lon'])
    print(datafilt.shape)
    print("Filtering....")
    for ll in range(len(lat)):
        tstrt = systime.process_time()
        print("latitude "+str(lat[ll].values))
        print("filtering current latitude")
        print(np.squeeze(data[:, ll, :].values).shape)
        datafilt2d = st.kf_filter(np.squeeze(data[:, ll, :].values), spd, tMin, tMax, kMin, kMax, hMin, hMax, waveName)
        print("write data for current latitude to array")
        datafilt[:, ll, :] = datafilt2d
        print(systime.process_time()-tstrt, 'seconds')
    print("Done!")
    print(datafilt.min(), datafilt.max())

    # save filtered data to file
    print("save filtered data to file")
    ds = xr.Dataset({'precipitation': datafilt}, {'time':time, 'lat':lat, 'lon':lon})
    ds.to_netcdf(pathout+fileout+".nc")
 

    

