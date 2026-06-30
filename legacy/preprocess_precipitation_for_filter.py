"""
preprocess_precipitation_for_filter.py

AUTHOR: Q. Lawton, NSF National Center for Atmospheric Research (NCAR)
DATE: 2024-01-15

This script preprocesses precipitation data for spatio-temporal filtering.
It loads model and observation data, slices it to the desired time and latitude range, and pads the data for further processing.
The script is designed to handle both model and observation data, with options to only process observation data if specified.
It outputs padded datasets for both the original and extended lengths, which can be used for further analysis or filtering.

"Original" length refers to the data as it is, while "extended" length includes additional padding with observation data to account for edge effects in filtering. 

Note that there are several hard-coded values in this script, such as the latitude range and the time resolution, which may need to be adjusted based on the specific dataset and analysis requirements.
"""

import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import sys

data_file_in = sys.argv[1] 
data_dir_out = '/glade/derecho/scratch/qlawton/model_data_for_filter_daily_mean/TP/'

obs_dir = '/glade/work/qlawton/DATA/IMERG/'
model_name = sys.argv[2]
print(model_name)
tres = int(sys.argv[3])
pad_num = int(30*(24/tres))
just_obs = sys.argv[4]
hr_sl = 12

lat_min = -10
lat_max = 20
### Load in the model data
if just_obs == 'True':
    data_xr = xr.open_dataset(data_file_in)
else:
    data_xr = xr.open_dataset(data_file_in).precipitation
###
st_date_str = datetime.strftime(pd.to_datetime(data_xr.time.values[0]) - timedelta(hours=12), '%Y%m%d%H')
data_name_out_start = model_name+'_'+st_date_str+'_'#'mpas_2021092400_'#model_name-inital_date-type_of_padding
obs_name_out_start = 'obs_for_'+model_name+'_'+st_date_str+'_'#mpas_2021092400_'
### Check the years to load in the data correctly
st_year = data_xr.time.dt.year.values[0]
end_year = data_xr.time.dt.year.values[-1]
file = obs_dir+'combined_1998_2024_daily_mean_IMERG_with_late_run.nc'
if end_year == st_year: #If the same year...
    obs_xr = xr.open_dataset(file).precipitation
    obs_xr = obs_xr.sel(time = obs_xr['time'].dt.hour == hr_sl,
                       lat = slice(lat_min, lat_max)).load()
else: #but if they are different, need to append
    obs_xr = xr.open_dataset(file).precipitation
    obs_xr = obs_xr.sel(time = obs_xr['time'].dt.hour == hr_sl,
                       lat = slice(lat_min, lat_max)).load()


data_st_tm = data_xr.time[0]
data_slc_st_tm = data_xr.time[0]
data_end_tm = data_xr.time[-1]
thirty_day_start = pd.to_datetime([data_st_tm.values]) - timedelta(days=30)
#thirty_day_start = data_st_tm - timedelta(days=30)

#print(data_end_tm)
#print(data_st_tm, data_end_tm)
### Slice out the data
obs_slice = obs_xr.sel(time=slice(data_st_tm, data_end_tm))
obs_prev_slice = obs_xr.sel(time=slice(thirty_day_start.to_pydatetime()[0], data_slc_st_tm))
obs_full_slice = obs_xr.sel(time=slice(thirty_day_start.to_pydatetime()[0], data_end_tm))
#print(obs_slice.timxe)
if just_obs != True:
    data_xr = data_xr.sel(time = data_xr['time'].dt.hour == hr_sl)
    data_xr = data_xr.sel(lat=slice(lat_min, lat_max))
    ### We want to load in the data, cut it down to the 12Z time, and append (minus the last time step)
    combined_slice = xr.concat([obs_prev_slice, data_xr.isel(time=slice(1, None))], dim = 'time')
    padded_combined_slice = combined_slice.pad(time=pad_num, constant_values = 0)
    padded_data_slice = data_xr.pad(time=pad_num, constant_values = 0)
padded_obs_slice = obs_slice.pad(time=pad_num, constant_values = 0)
padded_obs_prev_slice = obs_prev_slice.pad(time=pad_num, constant_values = 0)
padded_obs_full_slice = obs_full_slice.pad(time=pad_num, constant_values = 0)

### Data names
obs_out = data_dir_out+obs_name_out_start+'orig_length_padded.nc'
obs_ext_out = data_dir_out+obs_name_out_start+'extended_length_padded.nc'
data_out = data_dir_out+data_name_out_start+'orig_length_padded.nc'
data_ext_out = data_dir_out+data_name_out_start+'extended_length_padded.nc'

### Save data
padded_obs_slice.to_netcdf(obs_out)
padded_obs_full_slice.to_netcdf(obs_ext_out)
if just_obs != 'True':
    print('saving data')
    padded_data_slice.to_netcdf(data_out)
    padded_combined_slice.to_netcdf(data_ext_out)
