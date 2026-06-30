#!/usr/bin/env python
# coding: utf-8

"""
This script reads in data from a user specified file and filters it for
a specified region in wavenumber - frequency space.
User must set filename and path for input and output files, define start
and end time of the data, define the region to filter for, set the latitude
range to filter and specify the number of observations per day of the input
data.
This can be very slow if the data is high spatial and temporal resolution.
"""
import numpy as np
import xarray as xr
from tropical_diagnostics import spacetime as st
import time as systime
import sys

# file and pathname
pathin = sys.argv[3]
pathout = pathin+'FILTERED/'

filein = sys.argv[1]
t_res = int(sys.argv[2])
data_var = sys.argv[4]

print(filein)
### List of waves to loop over
wavelist = ['Kelvin', 'ER', 'MRG', 'MJO']
#wavelist = ['MRG'] #Re-run with a WH99 definition of MRGs, not the 2-6 bandpass from Gehne et al. (2022)
# number of obs per day
spd = int(24/t_res)

for wvi in range(len(wavelist)):
    waveName = wavelist[wvi]
    print(waveName)
    # filename foxr filtered data
    fileout = waveName+'_'+filein
    
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
    elif waveName == "MJO":
        # Filter for 30-96 day eastward
        tMin = 30
        tMax = 96
        kMin = 0
        kMax = 250
        hMin = -9999
        hMax = -9999
    elif waveName == "TD":
        # Matching method Rosi used in Rios-Berrios et al. (2023, JAMES)
        tMin = -2.5
        tMax = -12
        kMin = 6
        kMax = 250 #Super large to keep all big wavenumbers
        hMin = -9999
        hMax = -9999
    
    
    
    # read data from file
    print("open data set...")
    ds = xr.open_mfdataset(pathin+filein).transpose("time", "lat", "lon")
    data = ds[data_var].sel(lat=slice(latMin, latMax)).fillna(0).load()
    lat = ds['lat'].sel(lat=slice(latMin, latMax))
    lon = ds['lon']
    time = ds['time']


    ds.close()
    #print("done. size of data array:")
    if len(time) % 2 != 0: #if odd
        print('odd data')
        data = data.isel(time=slice(None, -1))
        time = time.isel(time=slice(None, -1))
    
    # filter each latitude
    datafilt = xr.DataArray(np.zeros(data.shape), dims=['time', 'lat', 'lon'])
    print("Filtering....")
    for ll in range(len(lat)):
        tstrt = systime.process_time()
        #print("latitude "+str(lat[ll].values))
        #print("filtering current latitude")
        datafilt2d = st.kf_filter(np.squeeze(data[:, ll, :].values), spd, tMin, tMax, kMin, kMax, hMin, hMax, waveName)
        #print("write data for current latitude to array")
        datafilt[:, ll, :] = datafilt2d
        #print(systime.process_time()-tstrt, 'seconds')
    print("Done!")
    #print(datafilt.min(), datafilt.max())

    # save filtered data to file
    print("save filtered data to file")
    ds = xr.Dataset({data_var: datafilt}, {'time':time, 'lat':lat, 'lon':lon})
    ds.to_netcdf(pathout+fileout)
    ds.close()

    

