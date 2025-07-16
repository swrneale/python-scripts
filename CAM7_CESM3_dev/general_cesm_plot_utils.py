''' UTILITIES AND FIGURE RELATED SUBROUTINES'''

import matplotlib.pyplot as mp


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


def plot_lat(ax,fig,zmp,vunits,cname,seas,mask_ocn,last_plot,params_latlon):

    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature # Map features

    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    
    pproj = ccrs.PlateCarree()

    # Plotting
    
    norm = cm.colors.BoundaryNorm(boundaries=levs_4colbar, ncolors=256)


    # Plotting
    
    pplot = ax.contourf(pvar.lon, pvar.lat, pvar, transform=pproj, cmap=cmap, extend=extend, **params_latlon)
    

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

    ax.set_title(fig_let[icase] + case_lname[icase] + rmse_text, fontsize=25)


















''' Set Parameters For Plotting '''

def set_params_lat(icase,obs_set):
    

    if ptype == 'lat':
    
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



''' Set Parameters For Plotting '''

def set_params_latlon(icase,ncases,case,cases,obs_set,lanom_plot):
    

      
# Selection for each variable
    
    if case =='PRECT':
        clevs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
        aclevs = [-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]
        cbar = 'PRGn'
        acbar = 'terrain_r'


# Only set to anomaly lpot if > first plot
    if lanom_plot and icase > 0:   lanom_case = True
    
    
    plot_opts.update({'levels':aclevs) if lanom_case else plot_opts.update({'levels':clevs)
    plot_opts.update({'cmap': acmap)   if lanom_case else plot_opts.update({'cmap': acmap}) 
            


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

  
#    if case == 'NVAP':
#        vari = 'PREH2O'
#        vscalex = 1.
#    if var in ['TAUX','TAUY'] and case in obs_cases:
#        vscalex = -1.



    return var_info





    
