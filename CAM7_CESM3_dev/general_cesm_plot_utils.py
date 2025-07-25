''' UTILITIES AND FIGURE RELATED SUBROUTINES'''

import matplotlib.pyplot as mp
import numpy as np

dir_fig = '/glade/u/home/rneale/python/python-figs/CAM7_CESM3_dev/'

''' Figure Routines '''













''' 1D Zonal Average '''

def plot_lat(ax,fig,zmp,vunits,cname,seas,mask_ocn,last_plot,params_lat):


    var = 'PRECT'
    fsize = 15
   
    fig_pref = 'test'
    tocean = ' - Ocean' if mask_ocn else ''

    
# A bit tricky but see how many black lines there are and change the symbol if it is not the first obs. case plotted

    omarker = ['8','v','+']
    nobs_lines = len([line for line in ax.lines if line.get_color() == 'black'])
    if nobs_lines: params_lat.update({'marker': omarker[nobs_lines]})
    
# Plot
    ax.plot(zmp['lat'], zmp, label=cname, markersize=10, markevery=3, **params_lat)



    
# Plot 1D zonal average
    
    if last_plot: 
    
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


def plot_latlon(ax,fig,icase,pvar,vunits,cname,seas,pregion, mask_ocn,lanom_plot,last_plot,params_latlon):

    from matplotlib import cm
    
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature # Map features
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    
    pproj = ccrs.PlateCarree()

    preg_area = get_pregion(pregion)

    lat_min = preg_area[pregion]['lat_min']
    lat_max = preg_area[pregion]['lat_max']
    lon_min = preg_area[pregion]['lon_min']
    lon_max = preg_area[pregion]['lon_max']
    
    # Plotting

    levs_4colbar = params_latlon['levels']
    norm = cm.colors.BoundaryNorm(boundaries=levs_4colbar, ncolors=256)


    # Plotting
    
    pplot = ax.contourf(pvar.lon, pvar.lat, pvar, transform=pproj, extend='both', **params_latlon)


    # Add color bar before solid countours
        
    if icase == 0 or (icase == 1 and lanom_plot) :
        cbar_area = [1.02, 0.05, 0.05, 0.85] ; cbar_orient = "vertical"
        cbar_area = [0.02, -0.2, 0.9, 0.05 ] ; cbar_orient = "horizontal"
        cbar = ax.inset_axes(cbar_area, transform=ax.transAxes)
        cbar = fig.colorbar(pplot, cax=cbar, orientation=cbar_orient)
        if icase ==0 : cbar.set_label(vunits)


    
    params_latlon.update({'cmap': None})
    pplot = ax.contour(pvar.lon, pvar.lat, pvar, transform=pproj, colors='black', linewidths=0.5, **params_latlon)


    # Mapping
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pproj)
    ax.coastlines(linewidth=2)
    ax.gridlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.add_feature(cfeature.BORDERS)
    
    # Add U.S. states
    if pregion == "US":
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="black")

    # Axes
    
    ax.set_yticks(np.arange(lat_min, lat_max + 10.0, 10.0), crs=pproj)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xticks(np.arange(lon_min, lon_max + 20.0, 20.0), crs=pproj)
    lon_formatter = LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

# Letter for fig nu=mber
    
    fig_let = '('+chr(97 + icase)+') '

    
# Mean/RMSE (of the subsetted pregion)

    pvar = pvar.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    gweights = np.cos(np.deg2rad(pvar['lat']))                                                                                                  
    gweights.name = "weights"       
    
    pvar_mean = pvar.weighted(gweights).mean(dim=["lat", "lon"])                                                                   
                                                                         
    pvar2 =  pvar ** 2
    pvar_rmse = np.sqrt(pvar2.weighted(gweights).mean(dim=('lat', 'lon')))

    fmean_text = f" [Mean: {pvar_mean:.3g}]"
   

    if lanom_plot and icase > 0:
         frmse_text = f" [RMSE: {pvar_rmse:.3g}]"
    else:
         frmse_text = ""

        
    ax.set_title(fig_let + cname + fmean_text + frmse_text, fontsize=15)


















''' Set Parameters For Plotting Lat Zonal Average '''

def set_params_lat(icase,case,obs_set):
    


    
    plot_cols = [
    "red",
    "royalblue",
    "darkorange",
    "forestgreen",
    "firebrick",
    "goldenrod",
    "mediumpurple",
    "deepskyblue",
    "crimson"]

# Select params

    plot_opts = {}

    
    
    if case in obs_set:
        plot_opts.update({'color': 'black'})
        plot_opts.update({'linestyle': '-'})
        plot_opts.update({'linewidth': 3})
        plot_opts.update({'marker': 'x'})
# Check it's not
        
    
    else:
        plot_opts.update({'color': plot_cols[icase]})
        plot_opts.update({'linestyle': '-'})
        plot_opts.update({'linewidth': 2})
        plot_opts.update({'marker': None})


    return plot_opts









''' Set Parameters For Plotting Lat-Lon'''

def set_params_latlon(icase,ncases,vname,obs_set,lanom_plot):
    
    plot_opts = {}
      
# Selection for each variable


    if vname =='PRECT':
        clevs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
        aclevs = [-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]
        cmap = 'terrain_r'
        acmap = 'PRGn'



# Only set to anomaly lpot if > first plot
    lanom_case = True if lanom_plot and icase > 0  else False
    
    
    plot_opts.update({'levels': aclevs}) if lanom_case else plot_opts.update({'levels': clevs})
    plot_opts.update({'cmap': acmap})   if lanom_case else plot_opts.update({'cmap': cmap}) 
            


    return plot_opts








''' Set variable Parameters'''    


def set_var_params (var_name,case):


    var_info = {}

    var_info['PRECT'] = {'vscale': 86400.*1000.,
                         'vunits':'mm/day',
                         'GPCP':(1.,'PRECT'),
                         'TRMM':(1.,'PRECT')
                        }

    var_info['LHFLX'] = {'vscale':1.,
                         'vunits':'W/m2',
                         'ERAI':(1.,'LHFLX'),
                         'LARYEA':(28.93,'QFLX'),
                         'WHOI':(1.,'LHFLX')

    
                        }

  
#    if case :: 'NVAP':
#        vari : 'PREH2O'
#        vscalex : 1.
#    if var in ['TAUX','TAUY'] and case in obs_cases:
#        vscalex : -1.



    return var_info




''' Grab Plot Region Domain '''


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
                'lon_min' : -90 , 'lon_max' : -30, 
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
                'lat_min' : -20 , 'lat_max' : 20,
                'lon_min' : 0., 'lon_max' : 359.,
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
    
            
        case 'Tropics': # Tropics Wide
            
             reg_info[pregion] = {
                'lat_min' : -45 , 'lat_max' : 45,
                'lon_min' : 0 , 'lon_max' : 360.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
            }


    return reg_info
















    
