"""
track_filtered_waves.py

AUTHOR: Q. Lawton, NSF National Center for Atmospheric Research (NCAR)
DATE: 2024-01-15

This script tracks filtered atmospheric wave systems (Kelvin, Tropical Depression, or Equatorial Rossby waves) using IMERG precipitation data.
It loads daily filtered precipitation data, applies wave tracking algorithms, and saves the tracked wave longitude and strength to a NetCDF file.
Usage:
    - Configure wave_type and input paths as needed.
    - Run the script to process data and output tracked wave systems.
Dependencies: xarray, numpy, pandas, scipy, datetime
"""

import xarray as xr
import numpy as np
from datetime import datetime
from scipy import signal
import pandas as pd

### INPUT PATHS
data_in_path = '/glade/work/qlawton/DATA/IMERG/DAILY_FILTERED_WITH_LATE/'
var_in_path = '/glade/work/qlawton/DATA/IMERG/DAILY_FILTERED_WITH_LATE/VAR/'

### DEFINITIONS

wave_type = 'ER' # 'Kelvin', 'TD', or 'ER'
N_or_S = 'S' ## Only for Rossby waves; track north or south of the equator

t_res = 24 #hours, resolution of dataset
if wave_type == 'Kelvin':
    speed_thresh = 30 #m/s
    day_cut = 3
    direction = 'eastward'
    lat_bin = [-10, 10]
    save_name = 'CCKW'
if wave_type == 'TD':
    speed_thresh = 20 #m/s
    day_cut = 3
    direction = 'westward'
    lat_bin = [0, 15]
    save_name = 'TD'
if wave_type == 'ER':
    speed_thresh = 15 #m/s
    day_cut = 3
    direction = 'westward'

    if N_or_S == 'N':
        lat_bin = [0, 10]
        save_name = 'ER_N'
    elif N_or_S == 'S':
        lat_bin = [-10, 0]
        save_name = 'ER_S'
    

test_save_path = '/glade/work/qlawton/CCEW_TRACKS/DAILY_IMERG_FULL_WITH_LATE/'
test_save_name = test_save_path+'padded_daily_mean_'+save_name+'_tracks_2001-2024_full_adj_seam_signal_init_1_cont_0.25.nc'

#### COMPILE FULL PATH
data_in = data_in_path+'daily_mean_padded_imerg-daily.'+wave_type+'.nc'
var_in = var_in_path+'daily_mean_var-daily-'+wave_type+'.nc'

#### Define the "speed limit" for connecting the waves
lon_m = 111*1000 #m in a degree longitude at the equator
m_limit = t_res*3600*speed_thresh #meters a wave can possibly travel west
lon_limit = m_limit/lon_m
back_allow = 5 #How many degrees backwards we allow connections...

##### FUNCTION DEFINITIONS
def run_CCKW_tracker(data_in, lon_limit_in, prominence=1, cont_threshold = 0.5, init_threshold = 0.5, hgt_in=0, extend_data = True, extension_size = 10, direction = 'westward'):
    thr_in = prominence
    lon_limit = lon_limit_in
    time_in_dt = pd.to_datetime(data_in.time)
    empty_stack = np.ones((1, np.shape(time_in_dt)[0]))*np.nan

    # if flip_lon == True:
    #     data_in = data_in.copy()
    #     data_in['longitude'] = -1 * data_in['longitude']
    
    for tm_i in range(len(time_in_dt)): #Loop over the data
        tm_raw_in = pd.to_datetime(data_in.time)[tm_i]
        tm_in = datetime.strftime(tm_raw_in, '%Y-%m-%d-%H')
        data_test = data_in.isel(time=tm_i).squeeze()
        
        if extend_data == True:
            data_new = np.concatenate((data_test[-extension_size:], data_test, data_test[:extension_size]))
        #### First, we will call the main ID script
            peaks_i = signal.find_peaks(data_new, height = hgt_in, prominence = thr_in)[0]
            valleys_i = signal.find_peaks(-data_new, height = hgt_in, prominence = thr_in)[0]
        else:
            peaks_i = signal.find_peaks(data_test, height = hgt_in, prominence = thr_in)[0]
            valleys_i = signal.find_peaks(-data_test, height = hgt_in, prominence = thr_in)[0]        
        
        if extend_data == True:
            peaks_i = [peak - extension_size for peak in peaks_i if extension_size<= peak < len(data_test)+extension_size]
            valleys_i = [valley - extension_size for valley in valleys_i if extension_size<= valley < len(data_test)+extension_size]
        
        lon_peak = data_test['longitude'].values[peaks_i]
        lon_valley = data_test['longitude'].values[valleys_i]
        data_peak = data_test.values[peaks_i]
        data_valley = data_test.values[valleys_i]
        
        #print(data_peak, data_valley)
        #### Get the longitudes and the VP values at these points
        
        if tm_i == 0: ## If we have our first step, we will simply just initiate our arrays

            #### These are our initial arrays
            act_lon = np.zeros((len(lon_valley), np.size(time_in_dt)))*np.nan
            sup_lon = np.zeros((len(lon_peak), np.size(time_in_dt)))*np.nan
            act_avgval = np.zeros((len(data_valley), np.size(time_in_dt)))*np.nan 
            sup_avgval = np.zeros((len(data_peak), np.size(time_in_dt)))*np.nan
            
                #### Now we actually put the values in the arrays (initialize everything in first time step)
            for wv in range(len(lon_valley)):
                act_lon[wv, tm_i] = lon_valley[wv]
                act_avgval[wv, tm_i] = data_valley[wv]
            # And that's it at first
        else: #But if we aren't just starting out, we have a little more work to do...
            used_prev_wv = []
            used_wv = []
            for prev_wv in range(act_lon.shape[0]):
                prev_wv_lon = act_lon[prev_wv, tm_i-1]
                if np.isnan(prev_wv_lon): #Obviously only want waves that exist at last time step
                    continue

                did_append = False
                #### Here's another special case: if within "lon_limits" of the right seam (the +180 mark), and we don't actually have a close match, we check otherside
                if direction == 'eastward':
                    cmp_bool1 = (180 - prev_wv_lon <= lon_limit)
                elif direction == 'westward':
                    cmp_bool1 = (prev_wv_lon - (-180) <= lon_limit)
                if cmp_bool1:
                    if direction == 'eastward':
                        possible = lon_valley >= prev_wv_lon
                    elif direction == 'westward':
                        possible = lon_valley <= prev_wv_lon
                    #print('Trying to append across')
                    if not any(possible): #if there are seemingly none to attach...

                        if direction == 'eastward':
                            dist_from_seam = 180 - prev_wv_lon
                            adjusted_dist = (180+np.min(lon_valley))+dist_from_seam
                        elif direction == 'westward':
                            dist_from_seam = prev_wv_lon - (-180)
                            adjusted_dist = (180 - np.max(lon_valley)) + dist_from_seam    
                        #print(adjusted_dist)
                        if adjusted_dist <= lon_limit: #and if the closest one on left side is within bounds...
                            if direction == 'eastward':
                                nearest_lon = np.min(lon_valley)
                                nearest_i = np.where(lon_valley == nearest_lon)[0][0]
                                nearest_dist = nearest_lon - prev_wv_lon
                            if direction == 'westward':
                                nearest_lon = np.max(lon_valley)
                                nearest_i = np.where(lon_valley == nearest_lon)[0][0]
                                nearest_dist = prev_wv_lon - nearest_lon
                                
                            nearest_avgval = data_valley[nearest_i]
                            
                            if nearest_i in (used_wv):
                                #### We will check to see if we appended the closest one. If not, we want to 
                                #### replaced the current one with the closest one, and previous the previous CCKW attachment
                                used_wv_loci = np.where(used_wv == nearest_i)[0][0]
                                other_wv_to_check = used_prev_wv[used_wv_loci]
                                if np.isnan(other_wv_to_check):
                                    continue
                                if direction == 'eastward':
                                    seam_cross = act_lon[other_wv_to_check, tm_i] < 0 and act_lon[other_wv_to_check, tm_i-1] > 0
                                elif direction == 'westward':
                                    seam_cross = act_lon[other_wv_to_check, tm_i] > 0 and act_lon[other_wv_to_check, tm_i-1] < 0   
                                if seam_cross:
                                    if direction == 'eastward':
                                        seam_dist = 180 - act_lon[other_wv_to_check, tm_i-1]
                                        new_dist = act_lon[other_wv_to_check, tm_i] + 180
                                        previous_dist = seam_dist + new_dist
                                    elif direction == 'westward':
                                        seam_dist = act_lon[other_wv_to_check, tm_i-1] - (-180)
                                        new_dist = 180 - act_lon[other_wv_to_check, tm_i]
                                        previous_dist = seam_dist + new_dist
                                else:
                                    if direction == 'eastward':
                                        previous_dist = act_lon[other_wv_to_check, tm_i] - act_lon[other_wv_to_check, tm_i-1]
                                    elif direction == 'westward':
                                        previous_dist = act_lon[other_wv_to_check, tm_i-1] - act_lon[other_wv_to_check, tm_i]
                                if np.abs(nearest_dist) < np.abs(previous_dist):
                                    act_lon[other_wv_to_check, tm_i] = np.nan
                                    act_avgval[other_wv_to_check, tm_i] = np.nan
                                    used_prev_wv[used_wv_loci] = np.nan
                                else:
                                    continue

                            if np.abs(nearest_avgval) < cont_threshold:
                                continue
                            used_prev_wv.append(prev_wv)
                            used_wv.append(nearest_i)

                            ### And of course, update the data
                            act_lon[prev_wv, tm_i] = nearest_lon
                            act_avgval[prev_wv, tm_i] = nearest_avgval
                            did_append = True                        
                            
                if direction == 'eastward':
                    possible = lon_valley >= prev_wv_lon
                elif direction == 'westward':
                    possible = lon_valley <= prev_wv_lon
                
                if any(possible) and did_append == False:
                    if direction == 'eastward':
                        nearest_lon = np.array(lon_valley)[possible].min()
                        nearest_dist = nearest_lon - prev_wv_lon
                    elif direction == 'westward':
                        nearest_lon = np.array(lon_valley)[possible].max()
                        nearest_dist = prev_wv_lon - nearest_lon
                    nearest_i = np.where(lon_valley == nearest_lon)[0][0]
                    
                    if nearest_i in (used_wv):
                        #### We will check to see if we appended the closest one. If not, we want to 
                        #### replace the current one with the closest one, and previous the previous CCKW attachment
                        used_wv_loci = np.where(used_wv == nearest_i)[0][0]
                        other_wv_to_check = used_prev_wv[used_wv_loci]
                        if np.isnan(other_wv_to_check):
                            continue
                        if direction == 'eastward':
                            seam_cross = act_lon[other_wv_to_check, tm_i] < 0 and act_lon[other_wv_to_check, tm_i-1] > 0
                        elif direction == 'westward':
                            seam_cross = act_lon[other_wv_to_check, tm_i] > 0 and act_lon[other_wv_to_check, tm_i-1] < 0   
                        if seam_cross:
                            if direction == 'eastward':
                                seam_dist = 180 - act_lon[other_wv_to_check, tm_i-1]
                                new_dist = act_lon[other_wv_to_check, tm_i] + 180
                                previous_dist = seam_dist + new_dist
                            elif direction == 'westward':
                                seam_dist = act_lon[other_wv_to_check, tm_i-1] - (-180)
                                new_dist = 180 - act_lon[other_wv_to_check, tm_i]
                                previous_dist = seam_dist + new_dist
                        else:
                            if direction == 'eastward':
                                previous_dist = act_lon[other_wv_to_check, tm_i] - act_lon[other_wv_to_check, tm_i-1]
                            elif direction == 'westward':
                                previous_dist = act_lon[other_wv_to_check, tm_i-1] - act_lon[other_wv_to_check, tm_i]

                        if np.abs(nearest_dist) < np.abs(previous_dist):
                            act_lon[other_wv_to_check, tm_i] = np.nan
                            act_avgval[other_wv_to_check, tm_i] = np.nan
                            used_prev_wv[used_wv_loci] = np.nan
                        else:
                            continue
                    #nearest_i is the element with a lon equal to the nearest lon...
                    nearest_avgval = data_valley[nearest_i]

                    
                    
                    if nearest_dist <= lon_limit:

                        if np.abs(nearest_avgval) < cont_threshold:
                            continue
                        ### Add to the list of used waves
                        used_prev_wv.append(prev_wv)
                        used_wv.append(nearest_i)

                        ### And of course, update the data
                        act_lon[prev_wv, tm_i] = nearest_lon
                        act_avgval[prev_wv, tm_i] = nearest_avgval

            #### Okay now that we are done, we now consider the waves we have left
            for wv in range(len(lon_valley)):
                #print(data_valley[wv])
                ### First, check the most recent
                if wv in used_wv: #If wave already exists, continue
                    continue
                else:
                    if np.abs(data_valley[wv]) < init_threshold: #If not strong enough to initiate...
                        continue
                    act_lon = np.vstack([act_lon, empty_stack])
                    act_avgval = np.vstack([act_avgval, empty_stack])

                    act_lon[-1, tm_i] = lon_valley[wv]
                    act_avgval[-1, tm_i] = data_valley[wv]
                ### Repeat for any crests remaining, now checking the timestep prior to this one 

                ### And finally, repeat a third time
    # if flip_lon == True:
    #     act_lon = -1 * act_lon
    return act_lon, act_avgval  

def clean_up(act_lon, act_avgval, day_cut, t_res):
    step_len = int(day_cut*(24/t_res))
    new_act_lon=[]
    new_act_avgval=[]
    for row in range(act_lon.shape[0]):
        test_in = act_lon[row, :][~np.isnan(act_lon[row,:])]

        if len(test_in)>=step_len: #If we have enough for a CCKW, we will preserve it
            if len(new_act_lon) == 0:
                new_act_lon = act_lon[row, :]
                new_act_avgval = act_avgval[row, :]
            else:
                new_act_lon = np.vstack([new_act_lon, act_lon[row,:]])
                new_act_avgval = np.vstack([new_act_avgval, act_avgval[row,:]])
    return new_act_lon, new_act_avgval

def connect_tracks(AEW_lon_enter, AEW_avgval_enter, direction = 'eastward'):
    AEW_lon_in = AEW_lon_enter.copy()
    AEW_avgval_in = AEW_avgval_enter.copy()
    st_list = []
    end_list = []
    changed_wv = []
    for row in range(AEW_lon_in.shape[0]):
        st_idx = np.where(~np.isnan(AEW_lon_in[row,:]))[0][0]        
        end_idx = np.where(~np.isnan(AEW_lon_in[row,:]))[0][-1]
        st_list.append(st_idx)
        end_list.append(end_idx)
    for i in range(len(st_list)):
        if end_list[i] in st_list: 
            paired_wv = np.where(end_list[i] == st_list)[0][0]
            if direction == 'eastward':
                lon_diff = AEW_lon_in[paired_wv, end_list[i]] - AEW_lon_in[i, end_list[i]] # end_list[i] is the time for both. i is the time for wave we look at , paired_wv the compare one
            elif direction == 'westward':
                lon_diff = AEW_lon_in[i, end_list[i]] - AEW_lon_in[paired_wv, end_list[i]]
            if np.abs(lon_diff) <= lon_limit and lon_diff>=-back_allow: #If this is truely a connecting wave case...
                averaged_point = np.mean([AEW_lon_in[paired_wv, end_list[i]],  AEW_lon_in[i, end_list[i]]])
                
                ### Change first wave
                AEW_lon_in[i, end_list[i]] = averaged_point
                AEW_lon_in[i, (end_list[i]+1):] = AEW_lon_in[paired_wv, (end_list[i]+1):] 
                
                ## Change second wave
                AEW_lon_in[paired_wv, end_list[i]] = averaged_point
                AEW_lon_in[paired_wv, :(end_list[i]+1)] = AEW_lon_in[i, :(end_list[i]+1)] 
                changed_wv.append([i, paired_wv])
    del_rows = []
    for i in range(len(changed_wv)):
        del_rows.append(changed_wv[i][0])
    AEW_lon_in = np.delete(AEW_lon_in, obj=del_rows, axis = 0)
    AEW_avgval_in = np.delete(AEW_avgval_in, obj=del_rows, axis = 0)
    return AEW_lon_in, AEW_avgval_in, changed_wv
                ### now we are going to modify input data so that we can cascade down changes if necessary


# #### START OF ACTUAL CODE


### LOAD IN THE FULL DATASET
print('Loading in data...')
raw_data_xr = xr.open_dataset(data_in)
var_data_xr = xr.open_dataset(var_in)

data_in_xr = raw_data_xr['precipitation']/np.sqrt(var_data_xr['precipitation'])

data_in_xr.coords['lon'] = (data_in_xr.coords['lon'] + 180) % 360 - 180
data_in_xr = data_in_xr.sortby(data_in_xr.lon)

### Turn into a Hovmoller
print('Averaging...')
hov_data_in = data_in_xr.sel(lat = slice(lat_bin[0], lat_bin[-1])).mean(dim='lat')

print(hov_data_in.shape)
### ADJUST INPUT DATA
data_in_xr = data_in_xr.rename({'lon':'longitude','lat':'latitude'})
hov_data_in = hov_data_in.rename({'lon':'longitude'})
hov_data_test = hov_data_in#.sel(time = slice('1980-01-01', '1980-12-31'))

### Adjust the "seam" of the data to be at 60W. This is a less active part of the globe, helping smooth out tracks.
longitude_arr = hov_data_test.longitude.values
longitude_arr = longitude_arr - 240 #Seam at 60W (240 degrees is +60W)

longitude_arr[longitude_arr < -180] = longitude_arr[longitude_arr < -180] + 360
hov_data_test['longitude'] = longitude_arr
hov_data_test = hov_data_test.sortby(hov_data_test.longitude)

### RUN CCKW IDENTIFICATION
print('Running initial tracker')
### NOTE: *-1 is hard coded to flip the sign of the data, as the CCKW tracker is designed to track positive anomalies using a trough method. So this flips data to identify valleys as peaks.
raw_act_lon, raw_act_avgval = run_CCKW_tracker(hov_data_test*-1, lon_limit, prominence = 0, cont_threshold = 0.25, init_threshold = 1, direction = direction)
print('connecting tracks')
int_act_lon, int_act_avgval, changed_wv_out = connect_tracks(raw_act_lon, raw_act_avgval, direction = direction)
print('cleaning up tracks')
final_act_lon, final_act_avgval = clean_up(int_act_lon, int_act_avgval, day_cut = day_cut, t_res = t_res)

print('Adjusting back to original longitude space')
# Adjust longitude values to shift the seam back to the original space.
# Longitudes less than or equal to -60 are shifted eastward by 240 degrees,
# while longitudes greater than -60 are shifted westward by 120 degrees.
# This logic ensures continuity across the artificial seam at 60W.

new_act_lon = final_act_lon.copy()
new_act_lon[final_act_lon <= -60] = final_act_lon[final_act_lon <= -60] + 240
new_act_lon[final_act_lon > -60] = final_act_lon[final_act_lon > -60] - 120

final_act_lon = new_act_lon
print('Creating XArray Object.')
### TURN INTO AN XARRAY OBJECT
time_array = hov_data_test.time
system_array = np.arange(0, final_act_lon.shape[0])+1

CCKW_track_xr = xr.DataArray(final_act_lon, 
                             dims = ["system", "time"], 
                             coords = dict(
                                 system=(["system"], system_array),
                                 time=time_array),
                                 attrs=dict(
                                     description="Longitude of Tracked "+save_name,
                                     units="deg")
                                 )
CCKW_str_xr = xr.DataArray(final_act_avgval, 
                             dims = ["system", "time"], 
                             coords = dict(
                                 system=(["system"], system_array),
                                 time=time_array),
                                 attrs=dict(
                                     description="Strength of Tracked "+save_name,
                                     units="standard deviation")
                                 )

CCKW_combined_xr = xr.merge([CCKW_track_xr.to_dataset(name=save_name+'_lon'), CCKW_str_xr.to_dataset(name=save_name+'_str')])

print('Saving Out')
### SAVE OUT
CCKW_combined_xr.to_netcdf(test_save_name)
