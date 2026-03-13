''' UTILITIES AND FIGURE RELATED SUBROUTINES'''

import xarray as xr
import matplotlib.pyplot as mp
import matplotlib.path as mpath


import numpy as np
import sys

dir_fig = '/glade/u/home/rneale/python/python-figs/CAM7_CESM3_dev/'

''' Figure Routines '''



















''' 1D Zonal Average '''

def plot_lat(ax,fig,icase,var,zmp,vunits,cname,seas,mask_ocn,lanom_plot,last_plot,params_lat):


    fsize = 15 # Default font size of all text.
   
    fig_pref = 'test'
    tocean = ' - Ocean' if mask_ocn else ''


# A bit tricky but see how many black lines there are and change the symbol if it is not the first obs. case plotted

    omarker = ['8','v','+','x']
    nobs_lines = len([line for line in ax.lines if line.get_color() == 'black'])
    if nobs_lines: params_lat.update({'marker': omarker[nobs_lines]})

# Plot (RHS axis if anom-plot

    if icase > 0 and lanom_plot :
    # Right axis

       if not hasattr(ax, "_twin"):   # first time, create it
          ax._twin = ax.twinx()
          ax._twin.set_ylabel("Difference from Control")
          ax._twin.set_ylim(-60., 20.)   # fixed constant range

       ax._twin.plot(zmp['lat'], zmp, label=cname,
                      markersize=8, markevery=5, **params_lat)

#        axr.set_ylabel("Difference from Control")
#        axr.set_ylim(-0.2,0.2)   # fixed range on right-hand axis

    else:
        zm_line, = ax.plot(zmp['lat'], zmp, label=cname, markersize=8, markevery=5, **params_lat)

# Get the color of the control line 

        cline_color = zm_line.get_color()
        if lanom_plot: # Plot Y-axis labels same color as 'control' line color if anom_plot
             ax.yaxis.set_tick_params(colors=cline_color)   




# Plot 1D zonal average

    if last_plot: 

# # RBN: Still need to figure out how to combine axes legend items with two axes.

#        lines1, labels1 = ax1.get_legend_handles_labels()
#        lines2, labels2 = ax2.get_legend_handles_labels()
#        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        mp.axhline(y=0, color='gray', linestyle='--', linewidth=2)
        mp.xlabel('Latitude',fontsize=fsize)
        mp.ylabel(var+' ('+vunits+')',fontsize=fsize)
        mp.title(f'Zonal Average of {var} - {seas} {tocean}',fontsize=fsize)
        mp.legend(fontsize=fsize)
        mp.grid(True)
        mp.tight_layout()
        
        tocean = '_ocn_' if mask_ocn else ''
        mp.savefig(dir_fig+fig_pref+'_zonal_ave_2d_'+var+'_'+seas+tocean+'.png', dpi=120, bbox_inches='tight')
        mp.show()






































''' 2D Lat-Lon Average '''


def plot_latlon(ax,fig,icase,var,pvar,vunits,cname,seas,pregion, mask_ocn,lanom_plot,last_plot,params_latlon):

    from matplotlib import cm
    
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature # Map features
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter



    preg_area = get_pregion(pregion)

    lat_min = preg_area[pregion]['lat_min']
    lat_max = preg_area[pregion]['lat_max']
    lon_min = preg_area[pregion]['lon_min']
    lon_max = preg_area[pregion]['lon_max']
    

    dpproj = ccrs.PlateCarree()
    
    if pregion in ['NPolar','SPolar']: 
        pproj = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70.0) # For Northern Hemisphere
        

        # ---- Create a circular boundary for Polar plots ----
        theta = np.linspace(0, 2 * np.pi, 100)
        center = np.array([0.5, 0.5])  # center of the axes
        radius = 0.5
        circle = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle_path = mpath.Path(circle * radius + center)
    
        ax.set_boundary(circle_path, transform=ax.transAxes)

    

    
    else:
        pproj = ccrs.PlateCarree()
        
        
        ax.set_yticks(np.arange(lat_min, lat_max + 10.0, 10.0), crs=pproj)
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
    
        ax.set_xticks(np.arange(lon_min, lon_max + 30.0, 30.0), crs=pproj)
        lon_formatter = LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)

  
    # Plotting

    levs_4colbar = params_latlon['levels']
    norm = cm.colors.BoundaryNorm(boundaries=levs_4colbar, ncolors=256)


    # Plotting (first enable NaNs to be masked out in gray)
    # -- Not sure if this is working yet --
    camp= params_latlon['cmap']
   
    cmapo = mp.get_cmap(camp).copy()
    cmapo.set_bad('lightgray')  # Plot light gray for NaNs
    params_latlon.update({'cmap': cmapo})

    pvarz = np.ma.masked_inside(pvar, -0.75, 0.75)

    pplot = ax.contourf(pvar.lon, pvar.lat, pvarz, transform=dpproj, extend='both', **params_latlon)


    # Add color bar before solid countours
        
    if icase == 0 or (icase == 1 and lanom_plot) :
#        cbar_area = [1.02, 0.05, 0.05, 0.85] ; cbar_orient = "vertical"
        cbar_area = [0.02, -0.2, 0.9, 0.05 ] ; cbar_orient = "horizontal"
        cbar = ax.inset_axes(cbar_area, transform=ax.transAxes)
        cbar = fig.colorbar(pplot, cax=cbar, orientation=cbar_orient)
        if icase ==0 : cbar.set_label(vunits)


    
    params_latlon.update({'cmap': None})
    
    pplot = ax.contour(pvar.lon, pvar.lat, pvar, transform=dpproj, colors='black', linewidths=0.5, **params_latlon)


    # Mapping
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=2,color='black')
    ax.gridlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='gray')


    # Add U.S. states
    if pregion == "US":
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="black")

  

# Letter for fig number

    fig_let = '('+chr(97 + icase)+') '
    fig_let = ""


# Mean/RMSE (of the subsetted pregion)


    pvar = pvar.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
   
    
    gweights = np.cos(np.deg2rad(pvar['lat']))                                                                                                  
    gweights.name = "weights"       
    
    pvar_mean = pvar.weighted(gweights).mean(dim=["lat", "lon"]).values   
                                                                         
    pvar2 =  pvar ** 2
    pvar_rmse = np.sqrt(pvar2.weighted(gweights).mean(dim=('lat', 'lon'))).values

    fmean_text = f" [Mean: {pvar_mean:.3g}]"
   

    if lanom_plot and icase > 0:
         frmse_text = f" [RMSD: {pvar_rmse:.3g}]"
    else:
         frmse_text = ""

    fall_text = ""
#    fall_text = fmean_text + frmse_text
    
    if cname == "GPCP": cname = "Observations"
    ax.set_title(fig_let + cname + fall_text, fontsize=20)



    if last_plot:
        fig.suptitle(var + " - " + seas, fontsize=20, y=0.95)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
#    mp.title(var+" - "+seas,fontsize=20)














''' 2D Lat-Pressure Average '''


def plot_latpres(ax,fig,icase,pvar,vunits,cname,seas,lanom_plot,last_plot,params_latpres):

    from matplotlib import cm
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter    
 
    # Plotting

    levs_4colbar = params_latpres['levels']
    norm = cm.colors.BoundaryNorm(boundaries=levs_4colbar, ncolors=256)


    # Plotting
#    pvar = pvar.sortby('lev', ascending=True)

    pplot = ax.contourf(pvar.lat,pvar.lev,pvar, **params_latpres)

    # Add color bar before solid countours
        
    if icase == 0 or (icase == 1 and lanom_plot) :
        cbar_area = [1.02, 0.05, 0.05, 0.85] ; cbar_orient = "vertical"
        cbar_area = [0.02, -0.2, 0.9, 0.05 ] ; cbar_orient = "horizontal"
        cbar = ax.inset_axes(cbar_area, transform=ax.transAxes)
        cbar = fig.colorbar(pplot, cax=cbar, orientation=cbar_orient)
        if icase ==0 : cbar.set_label(vunits)


    
    params_latpres.update({'cmap': None})
    pplot = ax.contour(pvar.lat, pvar.lev, pvar, colors='black', linewidths=0.5, **params_latpres)


    # Mapping
    


    
    # Axes
    lat_min = -90.
    lat_max = 90.
    
    ax.set_yticks(np.arange(100., 1000., 100.0)) 

    ax.set_xticks(np.arange(lat_min, lat_max+30., 30.0))
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lat_formatter)

    ax.invert_yaxis()

# Letter for fig nu=mber

    fig_let = '('+chr(97 + icase)+') '


# Mean/RMSE (of the subsetted pregion)

    gweights = np.cos(np.deg2rad(pvar['lat']))                                                                                                  
    gweights.name = "weights"       
    
    pvar_mean = pvar.weighted(gweights).mean(dim=["lat", "lev"]).values                                                        

                                                                         
    pvar2 =  pvar ** 2
    pvar_rmse = np.sqrt(pvar2.weighted(gweights).mean(dim=('lat', 'lev'))).values
    print(pvar_mean,pvar_rmse)

    fmean_text = f" [Mean: {pvar_mean:.3g}]"
   

    if lanom_plot and icase > 0:
         frmse_text = f" [RMSD: {pvar_rmse:.3g}]"
    else:
         frmse_text = ""

        
    ax.set_title(fig_let + cname + fmean_text + frmse_text, fontsize=15)












''' Set Parameters For Plotting Lat Zonal Average '''

def set_params_lat(icase,case,obs_set,nobs_sets):
    


    
    plot_cols = [
    "red",
    "royalblue",
    "darkorange",
    "forestgreen",
    "firebrick",
    "slateblue",
    "goldenrod",
    "pink",
    "deepskyblue",
    "crimson",
    "purple",
    "gray"]

# Select params

    plot_opts = {}

    
    if case in obs_set:
        plot_opts.update({'color': 'black'})
        plot_opts.update({'linestyle': '-'})
        plot_opts.update({'linewidth': 3})
        plot_opts.update({'marker': 'x'})
        
    
    else:
        plot_opts.update({'color': plot_cols[icase-nobs_sets]})
        plot_opts.update({'linestyle': '-'})
        plot_opts.update({'linewidth': 2})
        plot_opts.update({'marker': None})


    return plot_opts









''' Set Parameters For Plotting Lat-Lon'''

def set_params_latpres(icase,ncases,vname,obs_set,lanom_plot):
    
    plot_opts = {}

# Selection for each variable

    match (vname):

        case 'Q':
            clevs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
            aclevs = [-6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5,6]
            cmap = 'terrain_r'
            acmap = 'PRGn'

        case _:
            print('-Variable ',vname,' is not defined here')
#            sys.exit(0)


# Only set to anomaly lpot if > first plot
    lanom_case = True if lanom_plot and icase > 0  else False
    
    
    plot_opts.update({'levels': aclevs}) if lanom_case else plot_opts.update({'levels': clevs})
    plot_opts.update({'cmap': acmap})   if lanom_case else plot_opts.update({'cmap': cmap}) 
            


    return plot_opts



''' Set Parameters For Plotting Lat-Lon'''

def set_params_latlon(icase,ncases,vname,obs_set,lanom_plot):
    
    plot_opts = {}

# Selection for each variable


    match (vname):

### 2D VRAS ###
        case 'PRECT' | 'PRECC' | 'PRECL':    
            clevs = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20])*0.75
            aclevs = np.array([-12, -10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12])*0.75
            cmap = 'terrain_r'
            acmap = 'PRGn'

        case 'TS':    
            clevs = [17,18,19,20,21,22,23,24,25,26,27,27.5,28,28.5,29,29.5,30,30.5]
            aclevs = np.array([-4, -3, -2.5,-2, -1.5,-1, -0.5,0, 0.5,1, 1.5,2, 2.5,3, 4])
            cmap = 'RdBu_r'
            acmap = 'RdBu_r'
       
        case 'PCONVT':    
            clevs = [500,525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900]
            aclevs = np.array([-20.,-15, -10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 15,20])*20
            cmap = 'Blues_r'
            acmap = 'RdBu_r'

        case 'TAUX' | 'TAUY':    
            clevs = np.array([-12, -10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12])*0.01
            aclevs = np.array([-12, -10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12])*0.005
            cmap = 'RdBu_r'
            acmap = 'RdBu_r'

        case 'CLDLOW':
            clevs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
            aclevs = np.array([-10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10])
            cmap = 'terrain_r'
            acmap = 'RdBu_r'

        case 'SWCF':
            clevs = np.array([-200,-180,-160,-140,-120,-100,-80,-60,-40,-20,0.,20])
            aclevs = np.array([-10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10])*5.
            cmap = 'terrain_r'
            acmap = 'RdBu_r'

        
        case 'LWCF':
            clevs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90,100,120])
            aclevs = np.array([-10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10])*5.
            cmap = 'terrain_r'
            acmap = 'RdBu_r'


        case 'LHFLX' | 'SHFLX':
            clevs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])*2.
            aclevs = np.array([-10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10])*5.
            cmap = 'terrain_r'
            acmap = 'RdBu'
    
        case 'TMQ':
            clevs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90,100.])*0.5
            aclevs = np.array([-10., -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10])
            cmap = 'YlGnBu'
            acmap = 'BrBG'

# Aerosol species


#        case 'SFbc_a4':
#            clevs = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30,40,50])
#            aclevs = np.array([-50.,-40, -30., -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40,50])
#            cmap = 'terrain_r'
#            acmap = 'RdBu_r'

        case 'SFbc_a4' | 'SFpom_a4' | 'SFso4_a1' | 'pom_a4_CLXF' | 'pom_a4':

            clevs = np.array([5, 10, 15, 20, 25, 30,35,40,45,50,55,60])*10.
            aclevs = np.array([-50.,-40.,-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30,40,50.])*10.0

#            clevs = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
#            aclevs = np.array([-20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20])
            cmap = 'terrain_r'
            acmap = 'RdBu_r'



    

### 3D VARS ###
        case 'Q':
            clevs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
            aclevs = [-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10]
            cmap = 'terrain_r'
            acmap = 'PRGn'



    

        case _:
            print('- Variable ',vname,' is either not defined here, or is spelled wrong - no explicit cmap/acmap set')
#            sys.exit(0)

   

# Only set to anomaly lpot if > first plot
    lanom_case = True if lanom_plot and icase > 0  else False
    
    
    plot_opts.update({'levels': aclevs}) if lanom_case else plot_opts.update({'levels': clevs})
    plot_opts.update({'cmap': acmap})   if lanom_case else plot_opts.update({'cmap': cmap}) 
            


    return plot_opts









''' 
    Grab Plot Region Domain
'''

def get_pregion(pregion):

  
    
    nrow = None
    ncol = None

    reg_info = {}
    
    match pregion:
    
        case 'LabSea': # Labrador Sea
     
            reg_info[pregion] = {
                'lat_min' : 35 , 'lat_max' : 70,
                'lon_min' : 280 , 'lon_max' : 340,
                'plev_scale' : 0.2,
                'aplev_scale' : 0.2
            }
                
        case 'IO': # Indian Ocean 
            
             reg_info[pregion] = {
                'lat_min' : -10 , 'lat_max' : 35,
                'lon_min' : 50 , 'lon_max' : 120,
                'plev_scale' : 1.,
                'aplev_scale' : 1.
             }
    
        case 'US': # USA #
            
             reg_info[pregion] = {
                'lat_min' : 25 , 'lat_max' : 55,
                'lon_min' : -120 , 'lon_max' : -70, 
                'plev_scale' : 0.25,
                'aplev_scale' : 0.25
             }
    
        case 'SAm': # South America 
            
             reg_info[pregion] = {
                'lat_min' : -40 , 'lat_max' : 15,
                'lon_min' : 250 , 'lon_max' : 330, 
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
    
    
        case 'Aus': # Australia 
            
            reg_info[pregion] = {
                'lat_min' : -20 , 'lat_max' : 10,
                'lon_min' : 120 , 'lon_max' : 150,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
    
        case 'TP': # Tropical Pacific
            
             reg_info[pregion] = {
                'lat_min' : -10 , 'lat_max' : 10,
                'lon_min' : 120., 'lon_max' : 290.,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
    
             nrow = 5 ; ncol = 2 # Rows and columns
    
        case 'WP': # West Pacific
            
             reg_info[pregion] = {
                'lat_min' : -20 , 'lat_max' : 40,
                'lon_min' : 110 , 'lon_max' : 270.,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
            
        case 'IndoPac': # West Pacific
            
             reg_info[pregion] = {
                'lat_min' : -40 , 'lat_max' : 40,
                'lon_min' : 40 , 'lon_max' : 200.,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
            
        case 'Tropics': # Tropics Wide
            
             reg_info[pregion] = {
                'lat_min' : -45 , 'lat_max' : 45,
                'lon_min' : 0. , 'lon_max' : 360.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
            }

        
        case 'Boreal': # Tropics Wide
            
             reg_info[pregion] = {
                'lat_min' : 40 , 'lat_max' : 75,
                'lon_min' : 190. , 'lon_max' : 280.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
            }

    
        case 'NPolar': # Northern Polar (stereographic)
            
             reg_info[pregion] = {
                'lat_min' : 60. , 'lat_max' : 90.,
                'lon_min' : -180. , 'lon_max' : 180.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
            }
        
    
    return reg_info







''' LOGIC FOR DERIVED VARIABLES (interpolate: lat, lon, lev,), and aonamlies'''

def var_forplot(ds,var_in,var_name,var_save,icase,lanom_plot,is_obs,vproc):

    ndims = var_in.squeeze().ndim


# 1. Interpolate in the horizontal only if there are differences to be calculated

    print(' - Processing variable ready for plotting -')

# If nothing to be done then var_save remains none and var_plot = var_in

    var_plot_r = var_in 
    var_save_r = var_save
    
    if lanom_plot:
        
        if icase == 0:
            
            var_save_r = var_in
            
        else:
 
            var_plot_r = var_in.interp(lat=var_save.lat, lon=var_save.lon) # Use preexiting var_save as passed in
               
    

 # 2. Decide if we need to interpolate/vertical average/remap in the vertical (yes regardless of lanom_plot)
    
    if ndims > 2: # 3D

        # Set 3D var_save to be icase=0 var_in
        if icase == 0 and lanom_plot: 
                        
            if any(item in var_save_r.dims for item in ['lev','ilev']):  # 3D variable (from both obs and model).

                print(vproc)
                var_save_r = to_plevs(ds,var_save_r,popt=vproc)

#                if vproc == 'vint':
#                    var_save_r = to_plevs(ds,var_save_r,popt='pvint')

                    
                
        else:

            if any(item in var_in.dims for item in ['lev','ilev']):  # 3D variable (from both obs and model).

                var_plot_r = to_plevs(ds,var_in,popt='plevs')

                if vproc == 'vint':
                     var_plot_r = to_plevs(ds,var_in,popt='plevs')
    
    if lanom_plot and icase > 0:
    
        var_plot_r = var_plot_r - var_save # Subtracts the existing var_save for icase > 0


    
    

    return var_plot_r, var_save_r










'''
    CONSOLIDATE DATSET AND EXTRACT THE VARIABLES REQUIRED TO CONSTRUCT VARIABLE TO BE PLOTTED (THEY MAY NOT BE THE SAME)
'''

def getvar_derived(ds,var_need,seas):


# Select climo files (2 types - amwg/python versions) 
    if ds.sizes.get('time') == 12:
        mon_index = get_month_indexes(seas)
        ds = ds.sel(time=mon_index).mean(dim='time').squeeze()

    
    if var_need in ds: # Simple variable read (actual variable on file)
        var_in = ds[var_need]
        
    else: # Do something special if for e.g., PRECT=PRECC+PRECL, or U850 from U. 
        
        if var_need in 'PRECT' and 'PRECL' in ds and 'PRECL' in ds:
            var_in =  ds['PRECC']+ds['PRECL']


# Just trim time if ntime=1
    var_in = var_in.isel(time=0) if 'time' in var_in.dims else var_in

    

    return var_in, ds














''' Interpolate to pressure levels wither hybrid to pressure or pressure to different pressure '''


def to_plevs(ds,var_in,popt=None):


    from geocat.comp import interp_hybrid_to_pressure 
    import numpy.ma as ma

# Common pressure levels.

    grav = 9.81
    new_plevs = np.array([980,950., 925, 850, 800, 700, 600, 500, 400,300,200,100,50,10])*100.   # in Pa


# CESM hybrid to plev subset 

    if 'hyam' in ds:
        
       
        
        PS = ds['PS']               # surface pressure in Pa
        hyam = ds['hyam'] # Mid
        hybm = ds['hybm']

        hyai = ds['hyai'] # Interface
        hybi = ds['hybi']

        
        P0 = ds.attrs.get("P0", 100000.0)  # reference pressure in Pa

        
        PS = PS.squeeze('time', drop=True)

# Add in time dimension for the routine needs
        PS = PS.expand_dims('time')

        display(var_in)
        var_in = var_in.expand_dims('time')    

# Make sure dims are in right order.
        var_in = var_in.transpose('lev','time', 'lat', 'lon')

       
        
        PS = PS.transpose('time', 'lat', 'lon')


# ########## REGRID TO PRESSURE LEVELS ############

        if popt == 'plevs':

            print(' -Vertically average CAM dataset ')

# Call vinth2p to intepolate
            var_p =  interp_hybrid_to_pressure (
                data=var_in,
                ps=PS,
                hyam=hyam,
                hybm=hybm,
                p0=P0,
                new_levels=new_plevs,
                method='linear'          # 1 = linear interpolation in ln(p)
            )

# No assume we are going to interplate to a different pressure level set

# Unify pressure coordinate/name
            var_p = var_p.rename({'plev':'lev'}) # CHnage to lev
            var_p = var_p.assign_coords(lev=0.01*var_p.lev)



        if popt == 'pvint':

            print(' -Vertical integral of CAM dataset ')



# ########## VERTICAL INTEGRAL ###############

# Interface pressures: pint(time, ilev, lat, lon)
            pint = hyai * P0 + hybi * PS
            print(pint)

# Layer thickness in pressure: dp(time, lev, lat, lon)
            dp = pint.diff("ilev")  # matches lev

    


# Mass-weighted vertical integral
            var_p = (var_in * dp / grav).sum("lev", keep_attrs=True)





# Get rid of the added time coordinate in either case
        var_p = var_p.squeeze('time', drop=True)
        display(var_p)
    
    else:

        var_p= var_in.interp(lev=new_plevs, method='cubic')

    
    
    return var_p















'''
    Get season time indices.
'''

def get_month_indexes(month_or_season):

    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'xg': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    season_mapping = {
        'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11], 'ann': [1,2,3,4,5,6,7,8,9,10,11,12]
    }
    
    month_or_season = month_or_season.lower()

    
    if month_or_season in month_mapping:
        return [month_mapping[month_or_season]]
    elif month_or_season in season_mapping:
        return season_mapping[month_or_season]
    else:
        return None





'''
    Read in a land mask and apply to the input field.
'''

def apply_land_mask(ds,var_in):

# Is LANDFRAC in the current dataset (alsmot always for CAM output/climo files.)

    print(' - Apply a mask for land areas where LANDFRAC > 0.5')
    
    if 'LANDFRAC' in ds:
        
        print(' - Masking from LANDFRAC on current dataset grid')
        land_frac =  ds['LANDFRAC'].isel(time=0) if 'time' in ds['LANDFRAC'].dims else ds['LANDFRAC']
    
    else: # Use a generic mask and interpllate.

        lfrac_file =  '/glade/derecho/scratch/rneale/archive/f.cam6_4_089.FLTHIST_ne30.cam7.001/climo/f.cam6_4_089.FLTHIST_ne30.cam7.001_ANN_climo.nc'
        print(' - Using a default CAM LANDFRAC from')
        print('   -',lfrac_file)

        ds_mask = xr.open_dataset(lfrac_file,decode_times = False, chunks={})
           
        land_frac =  ds_mask['LANDFRAC'].isel(time=0) if 'time' in ds_mask['LANDFRAC'].dims else ds_mask['LANDFRAC']
        
        # Interpolate var_in to ds grid if needed
        same_lat = ds.lat.equals(ds_mask.lat)
        same_lon = ds.lon.equals(ds_mask.lon)

        if not (same_lat and same_lon):
            print(" - Lat and/or lon coordinates differ - interpolation required before masking")
            land_frac = land_frac.interp_like(var_in)

    land_mask = (land_frac < 0.2)
    var_masked = var_in.where(land_mask)   # keeps ocean only

    

    

    return var_masked



''' Set variable Parameters'''    


def set_var_params (ds,var_name,case):


# Check to see if variable is on the file

# Check to see if 3D when asking for 2D and vv.

# If on file then set some parameters

### 2D VARS ###
    var_info = {}

    var_info['PRECT'] = {'vscale': 86400.*1000.,
                         'vunits':'mm/day',
                         'GPCP':(1.,'PRECT'),
                         'TRMM':(1.,'PRECT')
                        }

    var_info['PRECC'] = {'vscale': 86400.*1000.,
                         'vunits':'mm/day',
                        }

                         
    var_info['TS'] = {'vscale': 1,
                      'voffset': -273.16,
                         'vunits':'$^o$C',
                         'HadISST_PI':(1.,'SST'),
                         'HadISST_PD':(1.,'SST')
                        }
    
    
    var_info['PCONVT'] = {'vscale': 0.01,
                         'vunits':'mb',
                        }

    var_info['LHFLX'] = {'vscale':1.,
                         'vunits':'W/m2',
                         'ERAI':(1.,'LHFLX'),
                         'LARYEA':(28.93,'QFLX'),
                         'WHOI':(1.,'LHFLX')
                        }
    var_info['SHFLX'] = {'vscale':1.,
                         'vunits':'W/m2',
                         'JRA25':(1.,'SHFLX'),
                         'LARYEA':(1.,'SHFLX'),
                        }

    
    var_info['TAUX'] = {'vscale': -1.,
                         'vunits':'N/m$^2$',
                         'MERRA':(1.,'TAUX'),
                         'JRA25':(1.,'TAUX'),
                         'ERS':(1.,'TAUX')
                       }
    
    var_info['TAUY'] = {'vscale': -1.,
                         'vunits':'N/m$^2$',
                         'MERRA':(1.,'TAUY'),
                         'JRA25':(1.,'TAUY'),
                         'ERS':(1.,'TAUY')    
                        }
    var_info['CLDLOW'] = {'vscale': 100.,
                         'vunits':'%',
                        }

    var_info['SWCF'] = {'vscale': 1.,
                         'vunits':'Wm$^-2$',
                         'CERES-EBAF':(1.,'SWCF'),
                         'CERES2':(1.,'SWCF'),
                         'ERBE':(1.,'SWCF'),
                        }

    var_info['LWCF'] = {'vscale': 1.,
                         'vunits':'Wm$^-2$',
                         'CERES-EBAF':(1.,'LWCF'),
                         'CERES2':(1.,'LWCF'),
                         'ERBE':(1.,'LWCF'),
                        }

    var_info['TMQ'] = {'vscale': 1.,
                         'vunits':'mm',
                         'NVAP':(1.,'PREH2O')
                       }
    
    var_info['AODVIS'] = {'vscale': 1.,
                         'vunits':'ND',
                        }




# ### 2D AEROSOL FIELDS ####

    var_info['SFbc_a4'] = {'vscale': 1.e13,
                         'vunits':'(kg/m$^2$/s)*1e13',
                        }

    var_info['SFpom_a4'] = {'vscale': 1.e13,
                      'vunits':'(kg/m$^2$/s)*1e13',
                        }
    
    var_info['SFso4_a1'] = {'vscale': 1.e13,
                         'vunits':'(kg/m$^2$/s)*1e13',
                        }
    
    var_info['pom_a4_CLXF'] = {'vscale': 1.e13,
                         'vunits':'(kg/m$^2$/s)*1e13',
                        }
    
    var_info['pom_a4'] = {'vscale': 1.e13,
                         'vunits':'(kg/m$^2$/s)*1e13',
                        }


# ## 3D VARS ###

    var_info['Q'] = {'vscale':1000.,
                         'vunits':'W/m2',
                         'MERRA':(1.,'Q')

                     
                        }
    

    


    return var_info







