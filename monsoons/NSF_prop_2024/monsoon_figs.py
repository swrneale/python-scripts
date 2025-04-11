import xarray as xr
import numpy as np

from matplotlib.cm import get_cmap
from matplotlib.dates import DateFormatter,MonthLocator
from matplotlib import cm
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cf
from cartopy.io import shapereader

import matplotlib.pyplot as mp
from scipy.interpolate import griddata
from geopy.distance import geodesic












'''
    Plot lat lon regions
'''
    
#fig, ax = mp.subplots(figsize=(20,12))

def plot_latlon_path(ds, vname, vscale,lat_points, lon_points, title="Path on Map"):


    levels = [k for k in range(1,15)]
    
# Grab variable to be plotted.

    da_in = ds[vname]
    da_in = da_in*vscale

    
# Plot Ranges
    margin = 5
    plon_min =  min(lon_points) - margin ; plon_max =  max(lon_points) + margin 
    plat_min =  min(lat_points) - margin ; plat_max =  max(lat_points) + margin

    lon2d, lat2d = np.meshgrid(da_in['lon'].sel(lon=slice(plon_min, plon_max)), da_in['lat'].sel(lat=slice(plat_min,plat_max)))
    
    da_pin = da_in.sel(lon=slice(plon_min, plon_max), lat=slice(plat_min, plat_max))

    
    mp.figure(figsize=(15, 9))

    # Use a PlateCarree projection (simple lat/lon)
    ax = mp.axes(projection=ccrs.PlateCarree())
    ax.set_title(title)

    # Add land and coastlines, restrict labeling to left and top
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False


    # Plot the path
   
    ax_cont = ax.contourf(lon2d, lat2d, da_pin, transform=ccrs.PlateCarree(),levels = levels,cmap='terrain_r',extend='max')
    ax.tick_params(top=False, right=False)
    ax.plot(lon_points, lat_points, '-o', color='blue', transform=ccrs.PlateCarree())
    
    
    
    
    # Annotate points (optional)
    for i, (lat, lon) in enumerate(zip(lat_points, lon_points)):
        ax.text(lon + -0.3, lat + 0.5, f'{i+1}', fontsize=12, transform=ccrs.PlateCarree())

    # Colorbar

    mp.colorbar(ax_cont)
  
    # Set map extent with some padding

    mp.show()




def plot_cross_section(ds, var_1d, var_2d, lat_points, lon_points):
    """
    Plot a cross-section (depth vs. distance) of a 3D variable along a lat-lon path.
    
    Parameters:
        ds : xarray.Dataset
            Dataset with variables [depth, lat, lon]
        var_name : str
            Name of the 3D variable to slice (e.g., 'temperature')
        lat_points, lon_points : list or array
            Lat/lon coordinates defining the path
    """
    # Get levs, assume same for all grid points
    levs = ds['lev'].values

    # Flatten 3D grid
    lon_grid, lat_grid = np.meshgrid(ds.lon, ds.lat)
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()

    # Create path distances
    distances = [0]
    for i in range(1, len(lat_points)):
        dist = geodesic((lat_points[i-1], lon_points[i-1]), (lat_points[i], lon_points[i])).km
        distances.append(distances[-1] + dist)
    distances = np.array(distances)

    # Interpolate data along the path
    var_interp = []
    for k, lev in enumerate(levs):
        layer = ds[var_2d].isel(lev=k).values
        points = np.column_stack((lat_flat, lon_flat))
        values = layer.flatten()
        interp_layer = griddata(points, values, (lat_points, lon_points), method='linear')
        var_interp.append(interp_layer)

    var_interp = np.array(var_interp)  # shape: (pressure, path_length)

    # Plot
    mp.figure(figsize=(15, 9))
 
    cs = mp.contourf(distances, levs, var_interp, levels=30, cmap='bwr')
    
    mp.gca().invert_yaxis()
    mp.colorbar(cs, label=var_2d)
    mp.xlabel('Distance along path (km)')
    mp.ylabel('Height (pressure)')
    mp.title(f'{var_2d.capitalize()} Cross-Section')
    mp.grid(True)
    mp.show()

