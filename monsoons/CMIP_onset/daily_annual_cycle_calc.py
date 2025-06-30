'''
 Utility routines for a number of CAM output processing
'''


import xarray as xr
import datetime as dt
import pandas as pd
import geopandas as gpd
import numpy as np
import cftime
import dask as dk

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.mpl.geoaxes
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormattr
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from cartopy.io import shapereader

#import regionmask as rmask
import geopandas

import matplotlib.pyplot as mp
from matplotlib.colors import ListedColormap

from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from affine import Affine

import glob as gb




'''
    CALCULATE DAILY CLIMATOLOGY FROM DAILY DATA 
'''




def return_common_loc(case_type,case_dir,run_name,var_name,time_freq):


# Re-trun common locations for datasets

    print('-- return_common_loc -- : Grabbing typical run path for '+case_type)

    
    if time_freq=='daily':

        if case_type == 'CMIP5':
            case_loc = case_dir+'/'+run_name+'_dmeans_ts_'+var_name+'.nc'  
            print(case_loc)
        
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
            
        if any(x in case_type for x in ['CAM7','CESM3']):
            case_loc = case_dir+'/'+run_name+'/tseries/'+run_name+'_clim_dmeans_ts_'+var_name+'.nc'
    
    return case_loc












'''
    CALCULATE DAILY CLIMATOLOGY FROM DAILY DATA
'''



def calc_daily_acycle(rname,cname,set_df,var_df):

# # Variable and Ensembles information.

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

# # Observed/then change var to the observed name

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

    
        run_names = sorted(gb.glob(run_wcard))

    
        ##
        ## Dataset Read: Climo read in  ##
        ##
    

        var_data = None
        
        if cname =='AIR':
    
                try: 
                    dset = np.loadtxt(run_names[0],skiprows = 3)
                except:
                    print(run_names[0],' not found')  
    
                print('-Dataset year Range = 19XX to 20XX')
    
                var_data = vscale*dset  
    
                # Set to xarray equivalent to other datasets
                var_data = xr.DataArray(var_data,dims="dayofyear", coords=[np.arange(1, 366, dtype=np.int64)])
        

        
        if any(x in cname for x in ['CAM7','CESM3']):    # Is already in dayofyear climo format.
            
                try: 
                    dset = xr.open_mfdataset(run_names)
                except:
                    print(run_names[0]+' not found')  
        
                print('-Dataset year Range = 19XX to 20XX')
        
                var_data = vscale*dset[var]
        
                # Rename time to be same as other cases
                var_data = var_data.rename({'time': 'dayofyear'})

                # Regional average.
                var_data = mask_data(var_data,rname,lon_lat)
            
                   
        
        

        if var_data is None:

        ##
        ## Dataset Read: Climo. Needs To Be Constructed ##
        ##


                try:
                    dset = xr.open_mfdataset(run_names)

                except:

                    print(run_names+' not found')



                ## Time period
                print('-Dataset year Range = ',dset.time[0].dt.year,' to ',dset['time'][:].dt.year)

                var_data = vscale*dset[var].sel(time=slice(years[0],years[1]))
                

                # Do some unit checking for vscale

                print('Mask Generated for ',cname)


    
## Regional land mask averge
                var_data = mask_data(var_data,rname,lon_lat)


# Gather all the day of years and average (could take a while)

                # Drop Feb 29 days if needed
                if len(var_data['time']) % 365 == 0:
                    var_data = var_data.groupby("time.dayofyear").mean()
                else:
                    da_no_feb29 = var_data.sel(time=~((var_data['time'].dt.month == 2) & (var_data['time'].dt.day == 29)))

                # 2. Group by day-of-year using a custom string
                    doy = da_no_feb29.time.dt.strftime('%m-%d')  # Ensures same format across years

                # 3. Group and average
                    var_data = da_no_feb29.groupby(doy).mean("time")
            

## Perform a cumulative sum through the average year.
        var_data = var_data.cumsum()



# Expand to send back array with each ensemble member

        if irun == 0:
            cam_acycle = var_data.expand_dims({'ens_num':nens})
        else:
            cam_acycle.load()
            cam_acycle[irun,:] = var_data.values
#	print(cam_acycle)
            
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

# read the Indian borders

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

def mask_data(da,reg_name,lon_lat):

    from rasterio.features import geometry_mask

    if reg_name == 'AIR':

        shapefile_path = '/glade/work/rneale/shapefiles/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'

#        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
#        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_boundary_lines_land.zip"

# Use geopandas to read directly from the URL (if your environment supports zip streaming)
#        world = gpd.read_file(f"zip://{url}")

        
#        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = gpd.read_file(shapefile_path)

        country = world[world.ADMIN == "India"]  # example: mask out Norway

        
#        country = country.to_crs("EPSG:4326")  # fallback, assuming lat/lon

        # Need to reverse lat sometimes if N->

        if da.lat[1]-da.lat[0] < 0:
            da = da.reindex(lat=list(reversed(da.lat)))
       
        # Create transform
        lon = da['lon'].values
        lat = da['lat'].values
        
                    
        res_lon = (lon[1] - lon[0])
        res_lat = (lat[1] - lat[0])


        
        transform = Affine.translation(lon[0] - res_lon / 2, lat[0] - res_lat / 2) * Affine.scale(res_lon, res_lat)
        
        # Step 5: Prepare output shape (lat, lon shape)
        ny, nx = len(lat), len(lon)
        

        
        # rasterio expects shapes in GeoJSON-like format
        mask = geometry_mask([geom.__geo_interface__ for geom in country.geometry],
                             out_shape=(ny, nx),
                             transform=transform,
                             invert=True)  # True to mask *inside* the shape

  
        
        if lat[0] > lat[-1]: mask = mask[::-1]
            
        # Now mask the values in the original array

        var_mask = da.where(mask).mean(dim=["lat", "lon"])
        print(var_mask)


#        var_mask.plot(edgecolor="black", facecolor="lightgreen")
#        mp.title("India - Natural Earth (110m)")
#        mp.axis("equal")
#        mp.show()

    else: 

        lonw,lone,lats,latn = lon_lat[0],lon_lat[1],lon_lat[2],lon_lat[3]
        
        var_mask = da.sel(lon=slice(lonw,lone),lat=slice(lats,latn)).mean(dim=('lat','lon'))



    return var_mask


#        if reg_name == 'AIR': ''' Quasi AIR mask'''

#                rr = rr.where((rr.lat < 36) & (rr.lat > 8) & (rr.lon > 68) & (rr.lon < 98))
# Hive off chunks od the lat-lon ractangle (does really capture the far Eastern part)
#                rr.loc[30:35,81:100] = np.nan
#                rr.loc[25:35,70] = np.nan
#                rr.loc[18:27,97] = np.nan
#                rr.loc[28,83:92] = np.nan
#                rr.loc[28:35,70:72] = np.nan

#                rr.sel(lat=slice(5.,37.),lon=slice(65.,100.)).plot(cmap='rainbow')




 


