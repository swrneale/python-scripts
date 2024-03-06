'''
	BLOCKING FIGURE ROUTINES
'''




'''
    block_plot_1d - Line plot of 1D blocking Z500 metric      

'''

import xarray as xr
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as mp

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geocat.viz as gv


import importlib
import sys
import pprint
import time


# Fig (png) output directory.
dir_fig = '/glade/u/home/rneale/python/python-figs/blocking/' 


###################################################################################################### 
#  Plot 1D blocking %age for each longitude and for each ensemble set (which includes observations)
######################################################################################################



def block_plot_1d (block_meta,ens_block_1d,bseason,pshade='1',fig_out=False):

    fname = '-> block_plot_1d -> '

    
    tstart = time.time()

    lone_offset = 90. ; dlon = 30. # Offset to 'roll' the xarray, and lon deg. spacing

    ens_names = list(block_meta.index)
    
    # Fig stuff
    fig, ax = mp.subplots(figsize=(20,10))
    
   
    # Plot line characteristicsf
    ens_cols = ['blue','red','green','purple'] ; imod=0
    obs_dash = ['-','--',':'] ; iobs =0
    obs_mark = ['o','s','+']  

    ens_ystarts =  block_meta['Start Year'].values
    ens_yends =  block_meta['End Year'].values

  

    
    # Loop ensemble sets
    
    for iens,ens_name in enumerate(ens_names):
        
        print(fname,'Plotting for ensemble',ens_name)
        
        ens_type = block_meta.loc[ens_name]['Ensemble Type']
        ens_nruns =  len(block_meta.loc[ens_name]['Run Name'])
        
        
        

        # Model vs. obs line settings.
        if ens_type=='model': ens_col = ens_cols[imod] ; ens_dash='-' ; ens_mark = None ; mark_size = None ; imod+=1
        if ens_type=='obs':   ens_col = 'black' ;        ens_dash = obs_dash[iobs] ; ens_mark = obs_mark[iobs] ; mark_size = 15 ;  iobs+=1

        # Do a deep copy as rpeated invocation of this routine for fine turning messes the original data up if I don't.
        da_iens = ens_block_1d[ens_name]

        
        # Shift lon of data for better regional plotting
        ilon_roll =  int(lone_offset/(da_iens.lon[1]-da_iens.lon[0]))
        da_iens = da_iens.roll(lon=ilon_roll)
            
        
        # Set rolling smoothing for display.
        da_iens = da_iens.rolling(lon=3,center=True).mean()
        da_iens = 100.*da_iens # Scale to %age

        #  Set min/mean/max of each ensemble set
        da_iens_ave = da_iens.mean(dim='name')

        
        # Shade betweeen options min/max range of +/- 1 or 2 std.
        
        if pshade=='mm':
            shade_title = 'min/max range'
            
            da_iens_min = da_iens.min(dim='name') 
            da_iens_max = da_iens.max(dim='name')
        else:
            std_mag = int(pshade)
            shade_title = '-/+ '+pshade+' std'
                      
            da_iens_std = da_iens.std(dim='name')     
            da_iens_min = da_iens_ave-std_mag*da_iens_std         
            da_iens_max = da_iens_ave+std_mag*da_iens_std       
  
        # Plot line and fill between min/max within ensemble
        plabel = ens_name if ens_nruns==1 else ens_name+'('+str(ens_nruns)+')'
     
        ax.plot(da_iens_ave.lon, da_iens_ave,lw=4,color=ens_col,linestyle=ens_dash,
                marker=ens_mark, markersize = mark_size, markevery=10, mew=3, fillstyle='none', label=plabel)   


        if (ens_nruns >1) : ax.fill_between(da_iens_ave.lon, da_iens_min, da_iens_max, alpha=0.35)


    
    

    mp.rcParams.update({'font.size': 22})
    
    ax.set_xlim([1,365]) 
    xticksn = np.arange(-lone_offset, 360-lone_offset+1, dlon)
    xticks = np.arange(0, 360+1, dlon)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(abs(tick))}°W' if tick < 0 else f'{int(tick)}°E' for tick in xticksn])
        
    ax.set_ylim([0.01,35])


    
# Add years into title if common.
    
    if (min(ens_ystarts) == max(ens_ystarts) and min(ens_yends) == max(ens_yends)): 
        yr_title = ' (yrs: '+ens_ystarts[0]+' - '+ens_yends[0]+')  '
        fig_mid_text = '_' + ens_ystarts[0]+ '-' + ens_yends[0]        
    else:
        yr_title = ' '
        fig_mid_text = ' ' 

    
        
    ax.set_title(bseason+ ' '+yr_title+'  '+shade_title)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Blocking Frequency (%)')

# Add year range into legend
    
    ax.legend()


    
    # Output figure
    
    if fig_out: 
        fig_mid_text = '_' + ens_ystarts[0]+ '-' + ens_yends[1]
        mp.savefig(dir_fig + 'block_1d_freq_' + "_".join(ens_names) + fig_mid_text + '_' +bseason+'.png',dpi=80,bbox_inches="tight")




#    mp.xticks(np.arange(-lone_offset, 271, 45), [f'{tick}°E' for tick in np.arange(-lone_offset, 271, 45)])


    print(fname,f'Duration: {time.time() - tstart}') ; print()

    return


def block_plot_2d():

    fname = '-> block_plot_2d -> '

    
    tstart = time.time()
    
    return

def block_plot_1d_pdf():
    
    return

def jet_var_plot():
    return