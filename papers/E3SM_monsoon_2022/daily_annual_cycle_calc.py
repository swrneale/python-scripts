'''
 Utility routines for a number of CAM output processing
'''


import xarray as xr
import datetime as dt
import pandas as pd
import numpy as np
import cftime
import dask as dk

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from cartopy.io import shapereader
import geopandas

import matplotlib.pyplot as mp
from matplotlib.colors import ListedColormap

from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import glob as gb




'''
    CALCULATE DAILY CLIMATOLOGY FROM DAILY DATA
'''




def return_common_loc(case_type,case_dir,run_name,var_name,time_freq):

    
# Re-trun common locations for datasets

    print('-- return_common_loc -- : Grabbing typical run path for '+case_type)
    if time_freq=='daily':
    
        if case_type == 'CESM1':
            case_loc = case_dir+'/'+var_name+'/'+run_name+'.cam.h1.*.nc'  

        if case_type == 'CESM2':
            case_loc = case_dir+'/'+var_name+'/'+run_name+'.cam.h1.*.nc'  
            
        if case_type == 'E3SMv2':
            case_loc = case_dir+'/'+run_name+'/atm/proc/tseries/day_1/'+run_name+'.eam.h1.'+var_name+'.18500101-20141231.nc'     

        if case_type == 'GPCP':
            case_loc = case_dir+'/'+run_name+'/GPCP_1DD_v1.2_199610-201407.nc'
        
        if case_type == 'TRMM':
            case_loc = case_dir+'/'+run_name+'/TRMM.PRECT.nc'

            
            
            
    return case_loc












'''
    CALCULATE DAILY CLIMATOLOGY FROM DAILY DATA
'''



def calc_daily_acycle(cname,set_df,var_df):

## Variable and Ensembles information.
        
    var = list(var_df.index)[0]

    cdir = set_df.loc[cname]['ens_dir'] 
    
    rpref = set_df.loc[cname]['ens_pref']
      
    rnums = set_df.loc[cname]['ens_rnums'] ; nens = len(rnums)
    years = set_df.loc[cname]['ens_years']

    vscale = var_df.loc[var]['vscale']  
    lon_lat = var_df.loc[var]['lon_lat']  
    
    if cname=='GPCP':
        lonw,lone,lats,latn = lon_lat[0],lon_lat[1],lon_lat[3],lon_lat[2]
    else:
        lonw,lone,lats,latn = lon_lat[0],lon_lat[1],lon_lat[2],lon_lat[3]
        
## Observed/then change var to the observed name

    obs_names = var_df.loc['PRECT']['osource']
    
    if cname in obs_names:         
        iobs = var_df.loc['PRECT']['osource'].index(cname)
        var = var_df.loc['PRECT']['onames'][iobs]
        vscale = var_df.loc['PRECT']['oscale'][iobs]
    
    print(cname,' - ',nens,' ensemble members')
    
  
    for irun in range(0,nens):
        
        print('')
        
        print('- Ensemble run # ',irun+1)
    
        this_run = rpref+rnums[irun]
    
        run_wcard = return_common_loc(cname,cdir,this_run,var,'daily')
        print(run_wcard)
        
        run_names = sorted(gb.glob(run_wcard))
    

## Dataset    
        try:
            dset = xr.open_mfdataset(run_names, chunks={'time': 1})    
                
        except:
            print(run_names+' not found')
        
## Time period
        var_data = vscale*dset[var].sel(time=slice(years[0],years[1]))
      
## Regional averge
        var_data = var_data.sel(lon=slice(lonw,lone),lat=slice(lats,latn)).mean(dim=('lat','lon'))
                                                 
## gather all the day of years and average                  
        var_data = var_data.groupby("time.dayofyear").mean()
                                
## Preform a cumulative sum through the average year.
        var_data = var_data.cumsum()
        
        
        
# Expand to send back array with each ensemble member
        
        if irun == 0:
            cam_acycle = var_data.expand_dims({'ens_num':nens})
        else:
            cam_acycle[irun,:] = var_data
            
    print('')
    
    return cam_acycle



'''
        PLOT SHADED REGION TO BE MASKED/AVERAGED

'''

def region_mask(ax1,lat_lon,lmask):
                
     # get country borders
        resolution = '10m'
        category = 'cultural'
        name = 'admin_0_countries'
        moffset = 10.
        
        lonw = lat_lon[0] ; lone = lat_lon[1]  
        lats = lat_lon[2] ; latn = lat_lon[3] 
        
        
        shpfilename = shapereader.natural_earth(resolution, category, name)

# read the shapefile using geopandas
        df = geopandas.read_file(shpfilename)

# read the german borders
        poly = df.loc[df['ADMIN'] == 'India']['geometry'].values[0]


#        figc = mp.figure(figsize=(10, 10))
#        ax1 = figc.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax1.coastlines('50m')
        ax1.add_geometries(poly, crs=ccrs.PlateCarree(), 
                  edgecolor='0.5',facecolor='gray')
        ax1.set_extent([lonw-moffset,lone+moffset, lats-moffset, latn+moffset], ccrs.PlateCarree())
  
        axins = inset_axes(ax_in, width="40%", height="40%", loc="upper right", 
                   axes_class=geoaxes.GeoAxes, 
                   axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
        axins.add_feature(cf.COASTLINE)


        
        return ax1
                


