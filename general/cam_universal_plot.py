#########################################

# CAM PROCESSING UTILS FUNCTIONS

#########################################

print('+++ IMPORTING UTILS FUNCTIONS +++')

### Imports ###
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import scipy as spy
import pandas as pd
import datetime as dt
import cftime as cft
from nc_time_axis import CalendarDateTime
import subprocess as sp	

import metpy.constants as mconst
import metpy.calc as mpc






'''

Lat Lon Countour Plots


'''

def latlon_countour:


	
# # Check #case and plot spaces.
ncases = case_desc.size
nplots = nrow*ncol

if ncases > nplots: 
    print('')
    print('*** nrow*ncol does not math ncases - exiting ***') 
    exit('nplot,ncase mi-match')

########## SET UP LAT LON PLOTS ##############

nseas = seas.size # Different for different seasons





# Grab/Process data
mscale = pvars_df.loc[var_name]['mscale']
lname = pvars_df.loc[var_name]['long name']
vunits = pvars_df.loc[var_name]['munits']
# Set up graphics
#pproj = ccrs.Mollweide(central_longitude=0.0)
pproj  = ccrs.PlateCarree()
tpproj = ccrs.PlateCarree()
#pproj = ccrs.EckertV(central_longitude=-180.)





fig,axs =  mp.subplots(nrows=nrow,ncols=ncol, constrained_layout=True,
                        subplot_kw={'projection': tpproj},
                        figsize=(28,12))
fig.patch.set_facecolor('white') # White not transparent background


#fig = mp.figure(figsize=(25, 16))

#axes_class = (GeoAxes,dict(map_projection=tpproj))

#axgr = AxesGrid(fig, 111, axes_class=axes_class,
#                    nrows_ncols=(nrow, ncol),
#                    axes_pad=axes_pad,
#                    share_all=share_all,
#                    cbar_location=cbar_location,
#                    cbar_mode=cbar_mode,
#                    cbar_pad=cbar_pad,
#                    cbar_size='2%',
#                    label_mode='')  # note the empty label_mode




# lat,lon Contouring choices #

#mp.close()

cmin  = pvars_df.loc[var_name]['cmin'] ; cmax = pvars_df.loc[var_name]['cmax']
acmin = pvars_df.loc[var_name]['acmin']; acmax = pvars_df.loc[var_name]['acmax']

dcont = (cmax-cmin)/ncint ; adcont = (acmax-acmin)/ancint

plevels =  np.arange(cmin,cmax+dcont,dcont,dtype=float)
aplevels = np.arange(acmin,acmax+adcont,adcont,dtype=float)

if var_name == 'PRECT':
#    plevels = np.array([1,2,3,4,5,8,10,12,15,20,25,30])
    plevels = np.array([1,2,3,4,5,6,8,10,12,14,16,20])
#    plevels = np.array([ip*0.25 for ip in plevels])
    aplevels = 0.5*np.array([-15,-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12,15])

if var_name == 'AODVIS':
    plevels = np.array([0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0,2.5])
    aplevels = np.array([-1.,-0.5,-0.2,-0.1,-0.05,-0.025,-0.01,-0.005,0.,0.005,0.01,0.025,0.05,0.1,0.2,0.5,1.])
    
 


############### SETUP ZONAL AVERAGE PLOTS ################

if lp3d_if_read_in:
    axzm_zm,axs_zm = mp.subplots(nrows=nrow, ncols=ncol,figsize=[16, 10])
    
# These array counters are just for the ax of the zm plots.
irow = np.zeros((nplots),dtype=int)
icol = np.zeros((nplots),dtype=int)

for ii in range(0,nrow): irow[ii*ncol:ii*ncol+ncol] = ii 
for ij in range(0,nrow): icol[ij*ncol:ij*ncol+ncol] = np.arange(0,ncol,1)

#### PLOTTING ###
    
print('+++++ Plotting +++++ '+var_name)
print('# cases = ',ncases)
            
pvar_ctrl = None










###################################
#### LOOP Cases and Subfigures ####
###################################
 
    
for iseas, this_seas in enumerate(seas):
    pvar3d_ctrl = None
    pvar_ctrl = None
    
    if nseas==1:  
        isub0 = 0 ; isub1 = ncases ; isub_step = 1 
    else :       
        isub0 = iseas ; isub1 = 2*ncases+iseas-nseas+2 ; isub_step = nseas 
    
    iplots = np.arange(isub0,isub1,isub_step)
    
    print('++++ SEASON = ',seas[iseas])
    
    
    axs = axs.flatten()
    for icase, ax in enumerate(axs):
        
        pvar = None # Reset
       
        cdesc = case_desc[icase] 
        cname = cases_df.loc[cdesc]['run name']
        mscale = pvars_df.loc[var_name]['mscale']   
        
        
        izm = irow[icase] ; jzm = icol[icase] # Paneling indices      
        
        if cdesc=='CE2':
            file_in = dir_root+cname+'/yrs_1979-2005/'+cname+'_'+this_seas+'_climo.nc'
        else :
            if cdesc in ['L58zm2new']:
                files_star = dir_root2+cname+'/'+cname+'*/'+cname+'*_'+this_seas+'_climo.nc' 
                file_in = glob.glob(files_star)[0]
                print(file_in)
            else:   
                file_in = dir_root+cname+'/0.9x1.25/'+cname+'_'+this_seas+'_climo.nc'
        
        if cdesc in ['GPCP','TRMM'] :
            file_in = dir_obs+cname+'_'+this_seas+'_climo.nc'
                
    
        print('')
        print('Case ',(icase+1),' of ',ncases,' -> '+cdesc+' - '+cname)
        print('-File = '+file_in)
        print('')
    
       
    
## Read in variable data ##   

        case_nc =  xr.open_dataset(file_in,engine='netcdf4',decode_times = False)
   
# Composite variables #
        if var_name == 'PRECT':
            if cdesc in ['TRMM','GPCP']:
                pvar = case_nc['PRECT']
                pvar = pvar.squeeze()
            else :
                pvar = case_nc['PRECC'].isel(time=0)+case_nc['PRECL'].isel(time=0) 
        
        
# Interpolated from 3D variable # 

        if l3d_var: 
        
            print('++ Interpolating from 3D field ##')
            pvar3d = case_nc[var_name_in].isel(time=0) # Read in 3D variable
#        pvar3d = units.Quantity(case_nc[var_name_in])
            pvar3d.squeeze()
    
                
            print('')
            pres_levu = plev_val * units.hPa # Set interpolation level (in hPa)
        
# Read in vars required for interpolation and 3D pressure values
            ps,P0,hyam,hybm =  case_nc['PS'].isel(time=0).squeeze(),case_nc['P0'],case_nc['hyam'],case_nc['hybm']
        
            pres_lev = pvar3d.copy(deep=True) # Make meta copy from the 3d variable
            pres_lev = hyam*P0 + hybm*ps # Change values to 3D pressure.
            pres_lev = units.Quantity(pres_lev, 'Pa') # Attach metpy units

# Perform vertical interpolation

            pvar = ps.copy(deep=True) # Copy off meta data from the 2D PS field.
      
            pvar = log_interpolate_1d(pres_levu, pres_lev, pvar3d, axis=0) # Annoying as all the meta data is lost.
            pvar = pvar.squeeze()
            pvar = xr.DataArray(pvar, coords=[pvar3d.lat, pvar3d.lon], dims=["lat", "lon"])
        
# Save off mean/ctrl
            if pvar3d_ctrl is None: pvar3d_ctrl = pvar3d
        
            if icase > 0 and ldiff_ctrl:
                pvar3d = np.subtract(pvar3d,pvar3d_ctrl)
                pvar3d = pvar3d.interp(lat=pvar3d_ctrl.lat,lon=pvar3d_ctrl.lon) if cdesc == 'C5' else pvar3d
        
# Standard 2D variable read.
        if pvar is None: 
            pvar = case_nc[var_name_in].isel(time=0).squeeze()
   
# Only scale the model data.
        if cdesc not in ['TRMM','GPCP'] : pvar = mscale*pvar # Scale to useful units.
 
        lon = case_nc['lon']    
        lat = case_nc['lat']
   
## Retain control case data ##   
        if pvar_ctrl is None: pvar_ctrl = pvar
   
## Contouring (full or anom) ##

        clevels = plevels   
        levs_4colbar = clevels
      
        cmap = pvars_df.loc[var_name]['cmap']      
       
    
## Modify if anom plot ##   
     
        if icase > 0 and ldiff_ctrl:
        
    
# I think CAM5 coords are slighly misalligned so have to interpolate to the pvar_ctrl (mostly CAM6/C6)
            pvar = pvar.interp(lat=pvar_ctrl.lat,lon=pvar_ctrl.lon) if cdesc == 'C5' else pvar
            pvar = np.subtract(pvar,pvar_ctrl)
            levs_4colbar = aplevels
            
            
            
            cmap = pvars_df.loc[var_name]['acmap'] 
       
        clevels = levs_4colbar[abs(levs_4colbar) > 1.e-15] # Remove zero contour, but sometimes it can be very small as well.
        
## Plotting for each ax ##  
# Domain, this seems flakey for cartopy ##
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pproj)
        ax.coastlines(linewidth=2)
        ax.gridlines()
        ax.add_feature(cartopy.feature.LAND, zorder=0)
        ax.add_feature(cartopy.feature.STATES, zorder=0)
        
#        for state in shpreader.Reader(states_shp).geometries():
            # pick a default color for the land with a black outline,
            # this will change if the storm intersects with our track
 #           facecolor = [0.9375, 0.9375, 0.859375]
 #           edgecolor = 'black'

 #           if state.intersects(track):
 #           facecolor = 'red'
 #           elif state.intersects(track_buffer):
 #           facecolor = '#FF7E00'

#            ax.add_geometries([state], ccrs.PlateCarree(),
#                    edgecolor='black',facecolor='pink')

        
        
#        ax.set_xticks(np.linspace(180., 180, 5), crs=pproj)
        

        ax.set_yticks(np.arange(lat_min, lat_max+10., 10.), crs=pproj)   
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
        
        
       
        ax.set_xticks(np.arange(lon_min, lon_max+10.,10.), crs=pproj)   
        lon_formatter = LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
            
## Add Cyclic Point ## 

        pvarc,lonc = cutil.add_cyclic_point(pvar,coord=pvar.lon)
        

## Translate to -180->+180 - maddening

#        ilon_pivot = int(lon.size/2)+1
#        pvarc = np.roll(pvarc, ilon_pivot, axis=1) # 144 points will roll you from 0-360 ->-180-180 data   
        
## CONTOUR PLOT (normalize colorbar for non-linear intervals) ##    
        norm = cm.colors.BoundaryNorm(boundaries=levs_4colbar, ncolors=256)
        pplot = ax.contourf(lonc, lat, pvarc,transform=pproj,cmap=cmap, extend='max',levels=clevels)
        
       
#### COLORBARS - Don't need one on every figure. #####  

#        cbar = mp.colorbar(pplot,shrink=0.75) # Plot individual color bars before line contour screws it up!    
#        cbar.ax.tick_params(labelsize=20)
  
#        pplot = mpl.colorbar.ColorbarBase(ax=ax, ticks=levs_4colbar[::2], # Every other tick.
#            cmap=cmap,boundaries=levs_4colbar,orientation='vertical',norm=norm)
        ax.tick_params(labelsize=15) 
        
        if ldiff_ctrl: # Make sure smallest contour intervals are white
            pplot = ax.contourf(lonc, lat, pvarc,levels=[-min(abs(clevels)),min(abs(clevels))], colors='w')
        
### Remove the colorbar entirely for these panels (just need two, 1 for full and 1 for anoms)
#        if icase in np.arange(1,ncases,1) and iseas==nseas-1:    
#            pplot = mpl.colorbar.ColorbarBase(ax=ax).remove()     
#            cbar.remove()
#           mp.subplots_adjust(right=0.9)
        
#### LINE PLOT  ####
        lpplot = ax.contour (lonc, lat, pvarc,transform=pproj
            ,levels=clevels,colors='black',linewidths=0.5)

        del(lonc) ; del(pvarc)
    
#### ADD PLOT TEXT/TITLE ####

        if icase==0 and nseas>1 :
            bbox=dict(boxstyle = "square",facecolor = "white")
            ax.text(-170., 40., seas[iseas],fontsize=30,fontweight='bold',bbox=bbox,zorder=10)
            
        if iseas==0:
            cdesc_fig = cdesc if not lcase_lname else case_lname[icase]
            ax.text(lon_min+2, lat_min+3., fig_let[icase]+cdesc_fig,fontsize=30,fontweight='bold',backgroundcolor = 'white',zorder=10)

            
    
    
    #### CONTOUR LABELING ####
       
#        mp.clabel(ax,
#            colors=['black'],
#            fontsize=8,
#            manual=False,  # Automatic placement vs manual placement.
#            inline=False,  # Cut the line where the label will be placed.
#            fmt=' {:.0f} '.format,  # Labels as integers, with some extra space.
#        )    

    ## ZM PLOTS ##

        if lp3d_if_read_in:
            pvar_zm = pvar3d.loc[:,lat_min:lat_max,lon_ave_w:lon_ave_e].mean(dim='lon') # Calc. zonal average
            lat_p = lat.loc[lat_min:lat_max]
            
#            print(axs_zm[izm,jzm].__dict__)
            if izm==nrow-1:
                axs_zm[izm,jzm].set_xlabel("Latitude", fontsize = 8) 
            if jzm==0:
                axs_zm[izm,jzm].set_ylabel("Pressure (mb)", fontsize = 8)
                
            axs_zm[izm,jzm].set_xticks(np.linspace(-90, 90, 7))   
            
            
            axs_zm[izm,jzm].set_title(fig_let[icase]+cdesc, x=0.72, y=0.02,fontweight ="bold",fontsize = 10,
                horizontalalignment='left',backgroundcolor = 'white')

            zm_plot = axs_zm[izm,jzm].contourf(lat_p, pvar_zm.lev, pvar_zm,levels=clevels,cmap=cmap,extend='both')

            fig_zm.colorbar(zm_plot,ax=axs_zm[izm,jzm],ticks=levs_4colbar[::2],pad=0.01)

            axs_zm[izm,jzm].tick_params(labelsize=8) # Change axis values font height. 
# Contouring
            zm_plot = axs_zm[izm,jzm].contourf(lat_p, pvar_zm.lev, pvar_zm,levels=[-1,1], colors='w')
            zm_plot = axs_zm[izm,jzm].contour(lat_p, pvar_zm.lev, pvar_zm,colors='black',linewidths=1.0,levels=clevels)
            
            axs_zm[izm,jzm].invert_yaxis()  
            axs_zm[izm,jzm].set_ylim([1000.,50.])

    ## Contour labaling ##
         
            ax.clabel(
                zm_plot,  # Typically best results when labelling line contours.
                colors=['black'], 
                fontsize=10,
                manual=False,  # Automatic placement vs manual placement.
                inline=True,  # Cut the line where the label will be placed.
                fmt=' {:.0f} '.format)  # Labels as lp3d, with some extra space.

   


#  Output figure
   
    
    
    
#    mp.subplots_adjust(wspace = .001)
    fig.suptitle(lname+' ('+vunits+')', fontsize=40)

    cbar_ax = fig.add_axes([1.02, 0.2, 0.01, 0.6])
#    mp.colorbar(pplot, ax=axs, cax=cbar_ax,location='right', shrink=0.6)
    print(dir(cbar_ax))
    mp.colorbar(pplot, cax=cbar_ax, orientation="vertical")
    cbar_ax.tick_params(labelsize=30)
#    mp.tight_layout()
    
    
    if nseas==1:
        mp.savefig('vres_FWscHIST__'+var_name+'_'+seas[0]+'_TRMM_Africa.png', dpi=50, bbox_inches='tight')  
    else :
        mp.savefig('vres_FWscHIST_'+var_name+'_TRMM_Africa.png', dpi=80, bbox_inches='tight') 

        
    mp.show()
#if l3d_var:
#    mp.savefig('CAM6_oview_zm_'+var_name_in+'_'+seas+'.png', dpi=300, bbox_inches='tight')