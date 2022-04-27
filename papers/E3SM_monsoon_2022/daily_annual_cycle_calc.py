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
import cartopy.mpl.geoaxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from cartopy.io import shapereader

import regionmask as rmask
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

        if case_type == 'AIR':
            case_loc = case_dir+'/'+run_name+'/AIR_daily_climatology.dat'
            
            
    return case_loc












'''
    CALCULATE DAILY CLIMATOLOGY FROM DAILY DATA
'''



def calc_daily_acycle(rname,cname,set_df,var_df):

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
    



    '''
           Loop Ensemble Members
    '''
        
        
    for irun in range(0,nens):
        
        print('')
        
        print('- Ensemble run # ',irun+1)
    
        this_run = rpref+rnums[irun]
    
        run_wcard = return_common_loc(cname,cdir,this_run,var,'daily')
        print(run_wcard)
        
        run_names = sorted(gb.glob(run_wcard))
        print(run_names)    

        ##
        ## Dataset Read: Climo read in  ##
        ##

        
        if cname =='AIR':

                try: 
                    dset = np.loadtxt(run_names[0],skiprows = 3)
                except:
                    print(run_names+' not found')  
                
                print('-Dataset year Range = 19XX to 20XX')
                
                var_data = vscale*dset  
                # Set to xarray equivalent to other datasets
                var_data = xr.DataArray(var_data,dims="time")

                
        else :
                
                
        

        ##
        ## Dataset Read: Climo. Needs To Be Constructed ##
        ##
        
             
        
                try:
                    dset = xr.open_mfdataset(run_names, chunks={'time': 1})    
                
                except:

                    print(run_names+' not found')
                
## Time period
                print('-Dataset year Range = ',dset['time'].min,' to ',dset['time'].max)
        
                var_data = vscale*dset[var].sel(time=slice(years[0],years[1]))
        
                print('HIIIIIIIII')
                reg_mask = gen_lsmask(rname,var_data.lon,var_data.lat)
                print('Mask Generated for ',)
                
                
         
        
        
## Regional land mask averge
                var_data = var_data.sel(lon=slice(lonw,lone),lat=slice(lats,latn)).mean(dim=('lat','lon'))
                                                 
## gather all the day of years and average                  
                var_data = var_data.groupby("time.dayofyear").mean()
        
    
        
        
                
        
## Perform a cumulative sum through the average year.
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

def region_mask(ax_in,lat_lon,lmask):
                
     # get country borders
        resolution = '10m'
        category = 'cultural'
        name = 'admin_0_countries'
        moffset = 10.
        
        lonw = lat_lon[0] ; lone = lat_lon[1]  
        lats = lat_lon[2] ; latn = lat_lon[3] 
        
        
        shpfilename = shapereader.natural_earth(resolution, category, name)
#
# read the shapefile using geopandas
        df = geopandas.read_file(shpfilename)

# read the german borders
#        print(df)
#        print()
        poly = df.loc[df['ADMIN'] == 'India']['geometry'].values[0]
#        print(poly)
#        figc = mp.figure(figsize=(10, 10))
#        ax_in = figc.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
#       

  
        axins = inset_axes(ax_in, width="15%", height="30%", loc="lower right", 
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                   axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
        
        
        axins.add_feature(cf.COASTLINE)
        axins.add_geometries(poly, crs=ccrs.PlateCarree(), edgecolor='0.5',facecolor='gray')
        axins.set_extent([lonw-moffset,lone+moffset, lats-moffset, latn+moffset], ccrs.PlateCarree())
        axins.coastlines('50m')

        
        
        return axins
                
'''
       Land mask for particular countries

'''

def gen_lsmask(reg_name,lon,lat):

#        lon = np.arange(-180, 180,1)
#        lat = np.arange(-90, 90,1)

#regionmask.defined_regions.natural_earth.countries_110.mask3d(lon,lat).plot()

#rrs = regionmask.defined_regions.natural_earth.countries_50
#rrs = regionmask.defined_regions
#dir(rrs)

#        rr = rmask.defined_regions.natural_earth.countries_110.mask(lon,lat)
        rr = rmask.defined_regions.natural_earth.countries_110[98].coords
        print(rr)
#dir(regionmask.defined_regions.natural_earth)

        if reg_name == 'India':

                rr = rr.where((rr.lat < 36) & (rr.lat > 8) & (rr.lon > 68) & (rr.lon < 98))
                print(rr)
# Hive off chunks
                rr.loc[30:35,81:100] = np.nan
                rr.loc[25:35,70] = np.nan
                rr.loc[18:27,97] = np.nan
                rr.loc[28,83:92] = np.nan
                rr.loc[28:35,70:72] = np.nan

                rr.sel(lat=slice(5.,37.),lon=slice(65.,100.)).plot(cmap='rainbow')
                
        return (lsmask)

#mask = regionmask.defined_regions.srex.map_keys.mask(lon,lat)
#rr = regionmask.defined_regions.srex.map_keys

      
#print(rr)
#rt = rr.mask(lon,lat)
#dir(rr)
#print(rr)
#rr.plot(add_label=False)
#rr = regionmask.defined_regions.natural_earth.countries_50.plot()
#dir(rr)


#rr.plot(add_label=False)
#land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
#dir(regionmask.defined_regions(add_label=False))
