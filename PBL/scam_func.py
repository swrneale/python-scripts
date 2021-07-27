#########################################

# SCAM PLOTING AND COMPUTE FUNCTIONS

#########################################

print('+++ IMPORTING MY FUNCTIONS +++')

### Imports ###
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import scipy as spy
import pandas as pd
import datetime as dt
import nc_time_axis
import subprocess as sp

import metpy.constants as mconst
import metpy.calc as mpc
import matplotlib.pyplot as mp
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

import os
import sys 

#from copy import deepcopy

### Constants ###

r_gas = mconst.dry_air_gas_constant.magnitude   # Specific gas constant for dry air (kg/joule)
cp_air = mconst.dry_air_spec_heat_press.magnitude # Specific heat for dry air
Lv = mconst.water_heat_vaporization.magnitude       # Latent heat of vaporization
grav = mconst.earth_gravity.magnitude

p0 = 100.*mconst.pot_temp_ref_press.magnitude # P0 but in Pa

r_cp = r_gas/cp_air    # r/cp







################################
#   1D tseries plotting     #
################################


def plot1d_ts_scam(rinfo):
	
    """
    1D TIMESERIES PLOTTING

    """
    
    
# Grab variable information for plotting
	
    
    plot1d_df = var_plot_setup("1d_ts",rinfo['Var List'])


## SPECIFIC LEGEND LOCATIONS

    vleg_left = ['PBLH','PBLH_DTH','PBLH_DQ'] # Vars. to put legend on the left not right.pvars_list = rinfo['pvars_list']_list

    ## Unbundle ##
    case_iop = rinfo['Case IOP']
    pvars_ts1d = np.array(rinfo['1dvars'])
    srun_names =  np.array(rinfo['Run Name']) # Has to be numpy so it can get appended
  
    sfile_nums = np.array(rinfo['File Num'])
    sfig_stub = rinfo['Stub Figs']
    dir_root = rinfo['Dir Root']
    pvars_list = rinfo['pvars_list']
    
    zoffset,iop_day,ttmin,ttmax = get_iop_info(case_iop)
    
    ## Derived vars.	
    ncases = srun_names.size

    
    
# Case an plotting info.


    print("")
    print("====================================")
    print(case_iop," - 1D PLOTTING VARIABLES")
    print("====================================")
    
    


    
    
    

    
    
#################
### LOOP VARS ###
#################

    
    
    for var in pvars_ts1d:
        
        print('================================================================================================')
        print('-- ',var,' ---- PLOTTING 1D TIME PLOTS ------>>>  ', plot1d_df.loc[var]['long_name'])
        
        
        ##
        fig1 = mp.figure(figsize=(16, 5))
        ax1 = fig1.add_subplot(111)
    
        
        
        vscale = plot1d_df.loc[var]['vscale'], ; ymin = plot1d_df.loc[var]['ymin'] \
            ; ymax = plot1d_df.loc[var]['ymax']

        var_les = plot1d_df.loc[var]['var_les'] 


    # Legend side        
        vleg_x = 0.22 if var in vleg_left else 0.97

        # Loop cases and plot
        for icase in range(0,ncases):
            pvar = None
            if (srun_names[icase] == 'LES' and var_les == ''): 
                continue

# GRAB Case data (SCAM or LES)
            scam_icase = scam_open_files(case_iop,sfile_nums[icase],srun_names[icase],dir_root)
        
            
## SCAM time and var
            if sfile_nums[icase] !='LES': 
                
                time = scam_icase.time
                
# The hour_frac time is for interpolation from LES to SCAM time.
                hour_frac = time.time.dt.hour+time.time.dt.minute/60.+zoffset
                hour_frac = hour_frac.values
                hour_frac = np.where(hour_frac<0,hour_frac+24.,hour_frac) # Makes continuous time when day goes into next day.
                
# Offxet time datetime object with the zoffset local time
                time = time + dt.timedelta(hours=zoffset) # Works for time axis and labeling
#                xhour_frac = time.dt.strftime("%H") 
            
## NEED TO MOVE ALL THIS BELOW INTO A FUNCTION ##

                if var == 'PRECT':
                    pvar = scam_icase['PRECC'].isel(lat=0,lon=0)+scam_icase['PRECL'].isel(lat=0,lon=0)
                        
                if var in ['PBLH_DTHL','PBLH_DQ','PBL_DQMAX']:  # PBL derived from d(thl/dz)

                    # Set up height instead of pressure
                  
                    plevm,zlevm = vcoord_scam('mid',scam_icase)
                    plevi,zlevi = vcoord_scam('int',scam_icase)


					# VARIABLE FOR GRADIENT #
                    if var in ['PBLH_DQ','PBL_DQMAX'] : pbl_var = 'Q'
                    
                    pvar = scam_icase[pbl_var].isel(lat=0,lon=0) # Variable dvardp
                    
                    
                    pvar['lev'] = zlevm[0,:].values # Add height instead of pressure for this variable (assume constant with time (i.e., PS(time) constant butBE CAREFUL)
    
                    
                    dvardz = pvar.differentiate("lev") # Find field gradient wrt HEIGHT!
#                   
            
                    dvardz.loc[:,100:] = 0.  # Restrict to a specific height region
                    dvardz.loc[:,:3000.] = 0.
                  
                    
                    dvardz_kmin = dvardz.argmin(axis=1) # Find the index of the maxium in the vertical		
                    dvardz_zmin = dvardz.lev[dvardz_kmin[:]] # Height level of max/min level.
                    
                    dvardz_ptop = dvardz.min(axis=1) # Actual value at min/max level.
                    
                    if var == 'PBL_DQMAX'  : pvar=dvardz_ptop*vscale # Scale for plotting
                    if var == 'PBLH_DQ' : pvar=dvardz_zmin # Scale for plotting

                    pvar.attrs['long_name'] = 'Height of max. Liq. Water Potential Temperature gradient'
                    pvar.attrs['units'] = 'm' 
						
      
                if pvar is None:
                    pvar = vscale*scam_icase[var].isel(lat=0,lon=0)

           
     

                ## LES time (map to SCAM time) - Assume preceded by SCAM case to interpolate to.
                
            if sfile_nums[icase] =='LES':
                

## NEED TO MOVE INTO A ROUTINE ##

                les_tstart = scam_icase['ts'] # Start time (local?) seconds after 00
                les_time = scam_icase['time'] # Time (from start?) in seconds

                les_toffset = 0. # Strange time stuff in LES
                hour_frac_les = (les_tstart+les_time)/3600.+les_toffset  # Transform into hours
                
                
                ## Specific quantity based on Q.
                if var in ['PBL_DQMAX']:
                    if var in ['PBLH_DQ','PBL_DQMAX'] : pbl_var = 'q'
                    data_les = scam_icase['q'] # Read in data # 
                    pvarp_sm = pvarp_sm.values.ravel()
                    plev = plev.values.ravel()
                    zlev = zlev.values.ravel()
                    hour_frac = np.repeat(hour_frac,np_les)
                    np_les = scam_icase.sizes['nz']
                    data_les = data_les*lscale

                    plev_les = scam_icase['p']
                    plev = 0.01*plev_les
                    zlev = scam_icase['zu']

                    pvar = data_les       
                

                if pvar is None: # Read in if special cases above not matched
                 
                    lscale = plot1d_df.loc[var,'lscale']
                    with xr.set_options(keep_attrs=True): 
                        pvar = scam_icase[var_les]
                    pvar = pvar*lscale
             
               

            # Funtion to interpolate to SCAM time.

                fles_int = spy.interpolate.interp1d(hour_frac_les.values,pvar,bounds_error=False)
                
            # Map to hour_frac time.
        
                pvar = fles_int(hour_frac)
                        
            # Convert to xarray Dataset for consistancy with SCAM
                           
                pvar = xr.DataArray(pvar)
                

           
## Merge back in for uniform plotting ##
            
                print(" --> ",sfile_nums[icase],' -- ',srun_names[icase], ' -- ymin/ymax --> ',  np.nanmin(pvar.values),np.nanmax(pvar.values)," <-- ",)
           
## Add to plot with the SCAM time coordinate


            ax1.plot(time,pvar)

    
    
#### #### #### #### #### #### 
##### END OF TIME LOOP ######
#### #### #### #### #### #### 
				            
    
    
    
    
    
    
# Observations

        plot_names = srun_names

        if case_iop == 'SAS' and var in ['PBLH_DTHL','PBLH_DQ']:
            
            ceil_obs   = [500.,300.,400.,400,500.,750.,1200.,1200.,1250.,1350.,1500.,1600.,1500.,1300.]
            ceil_obs_t = [5.,6,7,8,9,10,11,12,13,14,15,16,17,18]

            mp.plot(ceil_obs_t,ceil_obs,'+',color='black')
            plot_names = np.append(srun_names,"Ceilometer")

        # Axes stuff
        
        
## THESE ARE ALL OPTIONS FOR 'PRETTY' XAXIS INTERVALS (hrs for < 1day or days if > 1 day )
        
#        print(time.dt.day.values)
        npdays = (time.dt.day[-1].values-time.dt.day[0].values)
        
       
        if npdays <= 1: # Only hour plotting for 1 day or less.
            time_stride = mdates.HourLocator(interval = 1)  ;  myFmt = mdates.DateFormatter('%H') ##### EVERY HOUR
        else:
            time_stride = mdates.DayLocator(interval = 2)  ;  myFmt = mdates.DateFormatter('%D') ##### EVRY DAY
            
        mp.gcf().autofmt_xdate() # Diagonal xtick labels.
       
        ax1.xaxis.set_major_formatter(myFmt)
        ax1.xaxis.set_major_locator(time_stride)
        
        
### RES T IF LABELING ###
        
        ax1.set_ylim([ymin,ymax])
#        ax1.set_xlim(hour_frac,hours_frac)
        ax1.set_xlabel("Local Time (hr)")
        ax1.set_ylabel(plot1d_df.loc[var]['units'])
        ax1.set_title(plot1d_df.loc[var]['long_name'])


        ax1.legend(labels=plot_names, ncol=1, fontsize="medium",
            columnspacing=1.0, labelspacing=0.8, bbox_to_anchor= (vleg_x, 0.75),
            handletextpad=0.5, handlelength=1.5, borderaxespad=-5,
            framealpha=1.0,frameon=True)
        #        mp.show()
        mp.savefig(sfig_stub+'_plot1d_ts_scam_'+var+'.png',dpi=60)
        
        print('================================================================================================')
        print('')
        
        mp.show()
       
        print('')
        
#        mp.close()
















############################################
#  2D Time/Height Timeseries Plotting info. #
############################################ 


def plot2d_ts_scam(rinfo):

    
    	
    """
    2D TIMESERIES PLOTTING

    """
    
    

# Grab variable information for plotting and optional printing of info for all vars
	
    
    plot2d_df = var_plot_setup("2d_ts",rinfo['Var List'])  
    


    ### Vars that do not have -ve values in their full field.                             

    var_cmap0 = ['T','RELHUM','Q','CLOUD','THL','THV','WPRTP_CLUBB','WP2_CLUBB','WP3_CLUBB','THLP2_CLUBB']
    

    ## Unbundle ##
    case_iop = rinfo['Case IOP']
    pvars_ts2d = np.array(rinfo['2dvars'])
    srun_names =  np.array(rinfo['Run Name']) # Has to be numpy so it can get appended
    sfile_nums = np.array(rinfo['File Num'])
    dir_root = rinfo['Dir Root']
    sfig_stub = rinfo['Stub Figs']
    pvars_list = rinfo['pvars_list']

    zoffset,iop_day,ttmin,ttmax = get_iop_info(case_iop) # A few iop specific vars.

    
    
# Case and plotting info. print.

    print("")
    print("===================================")
    print(case_iop," - 2D PLOTTING VARIABLES")
    print("===================================")
   



# Display all available variables    

#    if (pvars_list) : print(plot2d_df) 

    
    ## Derived vars.	
    ncases = srun_names.size
    
    

    nclevs = 20 # Number of contour levels
    ppmin = 500. ; ppmax = 1000. # Pressure (mb) plot range
    zzmin = 0. ; zzmax = 3000.

    ptype = 'full' # Full/anom (case differening)/diff (time=0 differencing) 
    cmap_full = 'Purples'
    cmap_anom = 'RdYlBu_r'
#    cmap_anom = 'BrBG'

    ## TIME/HEIGHT PLOTTING ##

    ceil_obs   = [500.,300.,400.,400,500.,750.,1200.,1200.,1250.,1350.,1500.,1600.,1500.,1300.]
    ceil_obs_t = [5.,6,7,8,9,10,11,12,13,14,15,16,17,18]

    #######################
    #### VARIABLE LOOP ####
    #######################

    for var in pvars_ts2d:

        pvar = None


        ## VAR specific scaling and contour intervals ##

        vscale = plot2d_df.loc[var,'vscale'] ; cmin = plot2d_df.loc[var,'cmin'] ; cmax = plot2d_df.loc[var,'cmax']
        dcont = np.true_divide(cmax-cmin,nclevs,dtype=np.float)
        plevels = np.arange(cmin,cmax+dcont,dcont,dtype=np.float)

        acmin = plot2d_df.loc[var,'acmin'] ; acmax = plot2d_df.loc[var,'acmax']
        adcont = np.true_divide(acmax-acmin,nclevs,dtype=np.float)
        aplevels = np.arange(acmin,acmax+adcont,adcont,dtype=np.float)

        # Take out zero contour (annoying floating point problem!)
       
        aplevels[np.abs(aplevels) < 1e-10] = 0
    
        aplevels = aplevels[aplevels != 0]
        plevels = plevels[plevels != 0]
       

        ### First case plot (could be only plot) ###

        scam_icase = scam_open_files(case_iop,sfile_nums[0],srun_names[0],dir_root)
        
        plevm,zlevm = vcoord_scam('mid',scam_icase)
        plevi,zlevi = vcoord_scam('int',scam_icase)

        dzbot = 1000.*mpc.pressure_to_height_std(plevi[-1])
        scam_icase["time"] = scam_icase.time.assign_attrs(calendar="noleap")
       
#        scam_icase.time.attrs['calendar'] = 'noleap'
#        xr.decode_cf(scam_icase, decode_times=True)
        
        ### Derived Met variables ###
        if var in ['TH','THL','THV'] : 
            pvar = dev_vars_scam(var,scam_icase)
       
            
        ### All other vars. ###    
        if pvar is None :  # Set pvar if not already.
            pvar = scam_icase[var].isel(lat=0,lon=0).transpose()


        ### Determine Vertical Coord/Dim (lev/ilev) and time ###
        ## For scam needs to be 1 dimensional each for contourf
        plev = plevi[0,:] if 'ilev' in pvar.dims else plevm[0,:]
        zlev = zlevi[0,:] if 'ilev' in pvar.dims else zlevm[0,:]
        time = scam_icase.time
        

#        time_plot = time.time.dt.monotonic
        hour_frac = time.time.dt.hour+time.time.dt.minute/60.+zoffset
        print(hour_frac)
        hour_frac = hour_frac.values
        hour_frac = np.where(hour_frac<0,hour_frac+24.,hour_frac)

        print()
        print()
        print('================================================================================================')
        print('---- PLOTTING 2D TIME/HEIGHT PLOTS ------ >>>  ')
        print(' - ',var,' - ',pvar.attrs['long_name'],' -- cmin/cmax --> ',cmin,cmax)               
        print(' --> ', \
                    srun_names[0],' -- ',sfile_nums[0],' -- ', np.min(pvar.values),np.max(pvar.values))

##############             
# First plot #
##############

        fig1 = mp.figure(figsize=(16, 5))
        ax1 = fig1.add_subplot(111)
        pvar0 = vscale*pvar
      
        
        time = time + dt.timedelta(hours=zoffset) # Works for time axis and labeling

#        time = dt.datetime(time)
#        xtime = time.dt.strftime("%H") 
        
#        pcmap=cmap_full # First plot always full field and cmpa option
        pcmap=cmap_full if var in var_cmap0 else cmap_anom
#        pvarp = pvarp-pvarp[:,0] # Remove initial column values
       
        plt0 = ax1.contourf(hour_frac,zlev,pvar0,levels=plevels,cmap=pcmap, extend='both')   
        if ptype !='full' or ncases==1 : mp.colorbar(plt0)
    
        plt0 = ax1.contour(hour_frac,zlev,pvar0,levels=plevels,colors='black',linewidths=0.75)       
        ax1.clabel(plt0, fontsize=8, colors='black')
        
#        plt0 = ax1.contourf(hour_frac,zlev,pvar0,levels=[-min(np.abs(plevels)),min(np.abs(plevels))],colors='w') # Just to get white fill contours either side of zero
#        plt0 = ax1.plot(ceil_obs_t,ceil_obs,'X',color='red')
        mp.hlines(zlev, min(hour_frac), max(hour_frac), linestyle="dotted",lw=0.4)
        mp.suptitle(pvar.attrs['long_name']+(' - CLUBB' if 'CLUBB' in var else ' ')+' ('+plot2d_df.loc[var,'units']+')')

        ax1.set_title(srun_names[0])
        ax1.set_ylabel('Height (m)') 
        ax1.set_xlabel("Local Time (hr)")  
        #        ax1.set_ylim(ppmin, ppmax)
        ax1.set_ylim(zzmin,zzmax)
#        ax1.set_xlim(5, 17)  
        #        ax1.invert_yaxis()  
        
        # Change contouring for subsequent plots of diff/anom selected
        if ptype !='full': plevels=aplevels # Set to aplevels if anom fields

        # Color maps for negative valued filled.
       

        
            
#################################
# Loop for subsequent sub-plots #
#################################

        nn = len(fig1.axes)

        # LES CASE CANNOT BE FIRST CASE    

        for icase in range(1,ncases):

            pvarp = None

        # Open file (SCAM or LES) 
#            scam_icase = xr.open_dataset(sfiles_in[icase],engine='netcdf4')
            scam_icase = scam_open_files(case_iop,sfile_nums[icase],srun_names[icase],dir_root)

    
    
####### SCAM Specific #######   
    
            if sfile_nums[icase] != 'LES':

                plevm,zlevm = vcoord_scam('mid',scam_icase)
                plevi,zlevi = vcoord_scam('int',scam_icase)
                
    
                dzbot = 1000.*mpc.pressure_to_height_std(plevi[-1])
               
# Derived variable?    
                if var in ['TH','THL','THV'] : 
                    pvarp = dev_vars_scam(var,scam_icase)

                if pvarp is None :  # Set pvar if not already.
                    pvarp = vscale*scam_icase[var].isel(lat=0,lon=0).transpose()


        ### Determine Vertical Coord (lev/ilev) + time ###
                plev = plevi[0,:] if 'ilev' in pvar.dims else plevm[0,:]
                zlev = zlevi[0,:] if 'ilev' in pvar.dims else zlevm[0,:]

                zlev_line = zlev
                time = scam_icase.time
                hour_frac = time.time.dt.hour+time.time.dt.minute/60.+zoffset
                hour_frac = hour_frac.values
                hour_frac = np.where(hour_frac<0,hour_frac+24.,hour_frac)

                             
                
            # Remove initial column values (anom) or case0 (diff)
             
                if ptype == 'diff' : pvarp = pvarp-pvar0 ; pcmap = cmap_anom
                if ptype == 'anom' : pvarp = pvarp-pvarp[:,0] ; pcmap = cmap_anom
             

####### LES Specific #######

            if sfile_nums[icase] =='LES': 
                pvarp,hour_frac,zlev = get_les_dset(scam_icase,plot2d_df,var)
                pvarp = pvarp.transpose()

        ##### PLOTS #####                                                                
        ### Reshape sub-fig placements ###          

            nn = len(fig1.axes)

            for i in range(nn):
                fig1.axes[i].change_geometry(1, nn+1, i+1)
            ax1 = fig1.add_subplot(1, nn+1, nn+1)
        ####         
           

        #### Actual Plots ####

            print(' --> ', \
                    srun_names[icase],' -- ',sfile_nums[icase],' -- ', np.min(pvar.values),np.max(pvar.values))
            
            plt0 = ax1.contourf(hour_frac,zlev,pvarp,levels=plevels,cmap=pcmap, extend='both')
            mp.hlines(zlev, min(hour_frac), max(hour_frac), linestyle="dotted",lw=0.4) # Add level lines
  
                
        # Squeeze in colorbar here so it doesn't get messed up by line contours
                           
            if icase==ncases-1: 
#                mp.subplots_adjust(right=0.9)  
                mp.colorbar(plt0, cax=fig1.add_axes([0.92,  0.13, 0.02, 0.76]))
                
#            plt0 = ax1.contourf(hour_frac,zlev,pvarp,levels=[-min(np.abs(plevels)),min(np.abs(plevels))],colors='w') # Just to get white fill contours either side of zero
            plt0 = ax1.contour(hour_frac,zlev,pvarp,levels=plevels, colors='black',linewidths=0.75) 
            ax1.clabel(plt0, fontsize=8, colors='black')			   
#				ax1.clabel(plt0, fontsize=8, colors='black',fmt='%1.1f')

            stitle = srun_names[icase] if sfile_nums[icase] !='LES' else (srun_names[icase]+'-LES')
            ax1.set_title(stitle)
            ax1.set_xlabel("Local Time (hr)")
            ax1.set_ylim(zzmin,zzmax)
#            ax1.set_xlim(ttmin, ttmax)  
            
            

#            plt0 = ax1.plot(ceil_obs_t,ceil_obs,'X',color='red')
           

    #            ax1.yaxis.set_visible(False) # Remove y labels
    #            ax1.invert_yaxis()  

    ## Plot ##
    
        print('================================================================================================')
        print('')
        
        mp.savefig(sfig_stub+'_plot2d_ts_scam_'+var+'_'+ptype+'.png', dpi=80)    
        mp.show()

        print('')
        
        del pvar       

        
  




###############################################
# 1D Mean Height Plotting 
###############################################


def plot1d_mean_scam(rinfo):
    
    plot_mean_dic = {}
       
    plot_mean_dic['T']      = ['Temperature',1.,260.,305.,-20.,20.,'',1.,'K']
    plot_mean_dic['RELHUM'] = ['Relative Humidity',1.,10., 120.,-100.,100.,'',1.,'%']
    plot_mean_dic['CLOUD']  = ['Cloud Fraction',100., 0., 100.,-80.,80.,'',1.,'%']
    plot_mean_dic['Q']      = ['Specific Humidity',1000., 1., 12.,-5,5.,'q',1000.,'g/kg']










    
    
    
    
    



        
"""
###############################################
# 2D Snapshot/Height Timeseries Plotting info.
###############################################
"""


def plot1d_snap_scam(rinfo):
    
    
    	
    """
    1D TIMESTAMP COLUMN PLOTTING

    """
    
    
    
# Grab variable information for plotting
	
    
    plot_snap_df = var_plot_setup("1d_snap",rinfo['Var List'])

    

# Global stuff
   
    vleg_ul = ['TH','THL','THV'] # Vars with leg in upper left
    
    ppmin = 650. ; ppmax = 1000. # Pressure (mb) plot range    
    
    var_anim = 'THL'
    run_anim = '101c'
    pvar_anim = None
    pvars_snap = np.array(rinfo['snapvars'])


## Unbundle ##
    case_iop = rinfo['Case IOP']
    pvars_ts2d = np.array(rinfo['2dvars'])
    srun_names =  np.array(rinfo['Run Name']) # Has to be numpy so it can get appended
    sfile_nums = np.array(rinfo['File Num'])
    sfig_stub = rinfo['Stub Figs']
    tsnaps = rinfo['Snap Times']
    dir_root = rinfo['Dir Root']    
    pvars_list = rinfo['pvars_list']

    zoffset,iop_day,ttmin,ttmax = get_iop_info(case_iop)

 

    
    print("")
    print("=============================================")
    print(case_iop," - 1D SNAPSHOT PLOTTING VARIABLES")
    print("=============================================")

    

    
## Derived vars.	
    ncases = srun_names.size 
    ntsnaps = tsnaps.size
    
    
    
    
##########################
## VARIABLE LOOP #########
##########################
         
    for var in pvars_snap:

        long_name = plot_snap_df.loc[var,'long_name'] 
        vunits = plot_snap_df.loc[var,'units'] 
        cmin = plot_snap_df.loc[var,'cmin'] ; cmax = plot_snap_df.loc[var,'cmax']

        ### PLOT STUFF FOR SUBLOT SETUP ###

        fig1 = mp.figure(figsize=(16, 5))
        ax1 = fig1.add_subplot(111)
   

        # Plot several different functions...

        labelspacing = []
        labels = []
        
        
        print()
        print('================================================================================================')
        print('---- PLOTTING 1D HEIGHT SNAPSHOT PLOTS ------>>>  ')
        print(' - ',var,'  - ', plot_snap_df.loc[var]['long_name'],' --')
        
        
############################
## LOOP CASES AND SUBLOTS ##
############################

        nn = len(fig1.axes)

        for icase in range(0,ncases):
            
            pvar = None
            
#            scam_icase = xr.open_dataset(sfiles_in[icase],engine='netcdf4') 
            scam_icase = scam_open_files(case_iop,sfile_nums[icase],srun_names[icase],dir_root)
            # Vertical grid estimate.
            
            if sfile_nums[icase] != 'LES':
 
                pblh_dq =  pbl_grad_calc('Q',scam_icase)
#                pblh_dq =  pblh_dq.isel(lat=0,lon=0)
                pblh_dt =  scam_icase['PBLH'].isel(lat=0,lon=0)
    
                vscale = plot_snap_df.loc[var,'vscale'] 
 
                plevm,zlevm = vcoord_scam('mid',scam_icase)
                plevi,zlevi = vcoord_scam('int',scam_icase)
    
                dzbot = 1000.*mpc.pressure_to_height_std(plevi[-1])
            
                time = scam_icase.time
           
                hour_frac = time.time.dt.hour+time.time.dt.minute/60.+zoffset           
                hour_frac[hour_frac<0.] = hour_frac.values[hour_frac<0.]+24. # Reset if day ticks over to next day

            
## Get Data ##
                if var in ['TH','THL']: 
                    pvar = dev_vars_scam(var,scam_icase)
                    pvar = pvar.transpose()
           
                if pvar is None :  # Set pvar if not already.
                    pvar = scam_icase[var].isel(lat=0,lon=0)

          ### Determine Vertical Coord (lev/ilev) + time ###
                plev = plevi[0,:] if 'ilev' in pvar.dims else plevm[0,:]
                zlev = zlevi[0,:] if 'ilev' in pvar.dims else zlevm[0,:] 
                pvar = pvar*vscale
                    
               
            
            if sfile_nums[icase] == 'LES':
                pvar,hour_frac,zlev = get_les_dset(scam_icase,plot_snap_df,var)
                
                pblh_dq = scam_icase['zi_q'] # THis seems wrong compared to q field
                pblh_dt = scam_icase['zi_t'] # This seems right

                
### PRINT SOME CASE INFO ###

            print(' -->  ', \
                    srun_names[icase],' -- ',sfile_nums[icase],' -- ', np.min(pvar.values),np.max(pvar.values))

    
    
############################     
### LOOP SNAP SHOT TIMES ###
############################
        
    

            nplot_snaps = ntsnaps if sfile_nums[icase] != '106def' else ntsnaps-1
            for ii in range(0, nplot_snaps): 
                
#                print(' --- LOOP --- ')
#                print(tsnaps[ii])
#                print(hour_frac)
#                print(tsnaps)
                itt = np.min(np.where(hour_frac==tsnaps[ii])) # Plot at this time
                    
                pvart = pvar[itt,:]
              
                cmap=mp.get_cmap("tab10")   
                mp.plot(pvart,zlev,color=cmap(ii)) 
            
                ax1.set_xlabel(vunits)
                if icase ==0 : ax1.set_ylabel("meters")
                ax1.set_ylim(0, 3000.)
                ax1.set_xlim(cmin,cmax)
                
                         
# Pressure levele lines
                mp.hlines(zlev, cmin, cmax,lw=0.01) # plev horizontal lines
                
## PLOT PBLH DEPEDENT ON QUANTITY ###
                
               
                pblh = pblh_dq[itt] if var=='Q' else pblh_dt[itt] # Temp (theta) or Q criteria
               
                mp.hlines(pblh, cmin, cmax, linestyle="dashed",lw=1,color=cmap(ii)) # plot approx PBL depth
        
### Legend ###
                mp.suptitle(long_name+(' - CLUBB' if 'CLUBB' in var else ' ')+' ('+vunits+')')
 
                mp.title(srun_names[icase])
                lpos = 'upper left' if var in vleg_ul else 'upper right'
                mp.legend(labels=tsnaps, ncol=2, loc=lpos, 
                    columnspacing=1.0, labelspacing=1, 
                    handletextpad=0.5, handlelength=0.5, frameon=False)
               
### Reshape sub-fig placements (no need to do after last figure) ###
#            mp.show()
            
            if icase != ncases-1 :
    
                nn = len(fig1.axes)
           
                for i in range(nn):
                    fig1.axes[i].change_geometry(1, nn+1, i+1)
                ax1 = fig1.add_subplot(1, nn+1, nn+1)
             
# Save off variables/case for animation

       
        if var == var_anim and sfile_nums[icase] == run_anim: pvar_anim = pvar 
        
        

        print('================================================================================================')
        print()
    
        mp.savefig(sfig_stub+'_plot1d_snap_scam_'+var+'.png', dpi=80)    
        mp.show()
        
        del pvar # Reset pvar array
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# Animation
    if pvar_anim is not None:


        mp.cla()   # Clear axis
        mp.clf()   # Clear figure
        mp.close() # Close a figure window
#    fig1.close()


        print('+++ ANIMATION +++')
        print(pvar_anim)
        
        fig = mp.figure()
        ax = mp.axes(xlim=(0, 4), ylim=(-2, 2))
        line, = ax.plot([], [], lw=3)


### Add in animation for one of the cases ###
        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            x = np.linspace(0, 4, 1000)
            y = np.sin(2 * np.pi * (x - 0.01 * i))
            line.set_data(x, y)
#            line.set_data(pvar_anim[i,:], pvar_anim['lev'])
            return line,  
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=100, interval=40, blit=True)
    
        #mp.close(anim._fig)

        HTML(anim.to_html5_video())           

		
		
		
		
		
		
		
		
		
		

		
		
#########################################
# 1D/TIME ANIMATIONS
#########################################

        
        
        
        
def plot1d_anim_scam():
    
    nanim_vars = pvars_anim.size
    
#    fig, ax = mp.subplots(nrows=1, ncols=nanim_vars,figsize=(15, 5)))
   
    
# 
#    def init():
#        line.set_data([], [])
#        return line,
#    def animate(i):
#        x = np.linspace(0, 4, 1000)
#        y = np.sin(2 * np.pi * (x - 0.01 * i))
#        line.set_data(x, y)
#        return line,

#    line, = ax.plot(x, np.sin(x))
          
    for var in pvars_anim:

        pvar = None

        if var in ('TH','THL') : pvar = scam_in['T'].isel(lat=0,lon=0)*(0.01*p0/vplevs)**r_cp ; pvar.attrs['long_name'] = "Potential Temperature" ; pvar.attrs['units'] = "K" ; theta = pvar
        if var =='THL': pvar = theta-(theta/scam_in['T'].isel(lat=0,lon=0))*(Lv/cp_air)*scam_in['Q'].isel(lat=0,lon=0) ; pvar.attrs['long_name'] = "Liq. Water Potential Temperature"

            
        if pvar is None :  # Set pvar if not already.
            pvar = scam_in[var].isel(lat=0,lon=0)

        print('------ Animations ------>>>  ',var,' --- ',pvar.attrs['long_name'],' -- min/max --> ',  np.min(pvar.values),np.max(pvar.values))

    # Dynamically allocate subplots and animate Animate

#    ax.plot(pvar[0,:],vplevs) ;  ax.invert_yaxis()
    
  
    fig = mp.figure()
    
    
# Plotting frame.    
    ax = fig.add_subplot(111)
    ax.set_ylabel('mb') 
    for ip in range(1,5):
        n = len(fig.axes)
        print(n)
        for i in range(n):
            fig.axes[i].change_geometry(1, n+1, i+1)
        ax = fig.add_subplot(1, n+1, n+1)   
        
    def animate(ii):
        print(n)
        for i in range(n):
            ax[n].plot[i](pvar[ii,:],vplevs) 
    
#        del fig.axes
#        mp.show()
#        return ax
        
        
#    def animate(ii):
#        print(len(fig.axes))
#        ax = fig.add_subplot(111)
#        
#        ax.set_ylabel('mb') 
#        ax.plot(pvar[ii,:],vplevs) ;  ax.invert_yaxis()
#        for ip in range(1,2):
#            n = len(fig.axes)
#            print("hi2",n)
#            for i in range(n):
#                fig.axes[i].change_geometry(1, n+1, i+1)
#            ax = fig.add_subplot(1, n+1, n+1)
#            ax.plot(pvar[ii,:],vplevs) ;  ax.invert_yaxis()
#        del fig.axes
#        mp.show()
#        return ax
    print(fig.axes)
    animate(0)
    mp.show()
    
#    print(fig)
#    animate(1)
#    print(fig)
#    animate(200)
        
#
#       def init():
#        line.set_data([], [])
#        return line,     
    
#       def animate(i):
#        x = np.linspace(0, 4, 1000)
#        y = np.sin(2 * np.pi * (x - 0.01 * i))
#        line.set_data(x, y)
#        return line, 
#        for ip in range(pvars_anim.size-1):
#            n = len(fig.axes)
#            for i in range(n):
#                fig.axes[i].change_geometry(1, n+1, i+1)
#            ax = fig.add_subplot(1, n+1, n+1)
#            ax.plot(pvar[0,:],vplevs) ;  ax.invert_yaxis()
        
        
    anim = FuncAnimation(fig, animate, frames=np.arange(1,10))
#    print(anim)
    #    mp.show()
    mp.show()
    HTML(anim.to_html5_video())

    del pvar
    
   







##############################
######### ANIMATION ##########
##############################

def animation_test():

#        print(pvar_anim)
	fig = mp.figure()
	ax == mp.axes(xlim=(0, 4), ylim=(-20, 20))
	line, = ax.plot([], [], lw=3)

### Add in animation for one of the cases ###
	def init():
		line.set_data([], [])
		return line,
	def animate(i):
		x = np.linspace(0, 4, 1000)
		y = 0.1*i*np.sin(np.pi * (x*0.1*i - 0.1 * i))**0.5
		line.set_data(x, y)
		line.set_data(pvar_anim[i,:],pvar_anim['lev'])
		return line,  

	anim = FuncAnimation(fig, animate, init_func=init,
						frames=100, interval=40, blit=True)
	mp.close(anim._fig)
	HTML(anim.to_html5_video())   
	
	
	
	
	
	
	
	
	
	
	
	
	
######################################################
### PBL Height Calculations Based on q/th gradient ###
######################################################
	
	
def pbl_grad_calc(var_grad,scam_in):
	
# Var grad -> variable to be used gradient (Q or TH(eta))
    pres,zhgt = vcoord_scam('mid',scam_in)
    var_in = scam_in.T*(p0/pres)**r_cp if var_grad in 'TH' else scam_in.Q

# Gradient wrt height not pressure
    var_in['lev'] = zhgt[0,:].values # Add height instead of pressure for this variable (BE CAREFUL)           
    dvardz = var_in.differentiate("lev") # Find field gradient wrt HEIGHT (over limited region)	
    dvardz = dvardz.isel(lat=0,lon=0)# Find field gradient wrt HEIGHT (over limited region)	    

# Locate gradient maximum ain a bounded regin            
    ztop_mask = 3000 ; zbot_mask = 100  # Restrict to approx region of PBL top
    dvardz = dvardz.where((dvardz.lev < ztop_mask) & (dvardz.lev > zbot_mask)) # MASK IN LOCAL PBL REGION
    dvardz_kmin = dvardz.argmax(axis=1) if var_grad in 'TH' else dvardz.argmin(axis=1) # Find the index of the max/min in the vertical

# Map to a height level.   
    var = zhgt.isel(lev=dvardz_kmin)
  
# For compatability with more scripts   
    if 'lat' in list(var.dims): var = var.isel(lat=0,lon=0)

    return (var)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
######################################################
###  Vertical Coordinates From SCAM                ###
######################################################	
	    
# Grabs vertical profile of pressure interfaces and mid-levels and converts them to height for plotting
    
    
def vcoord_scam(imlev,scam_in):

 
    
    plevm = scam_in['hyam']*p0 + scam_in['hybm']*scam_in['PS'] # Mid level
    plevi = scam_in['hyai']*p0 + scam_in['hybi']*scam_in['PS'] # Interface level
    
 # For compatability with more scripts   
    if 'lat' in list(plevm.dims):
        plevm = plevm.isel(lat=0,lon=0)
        plevi = plevi.isel(lat=0,lon=0)
    
    plevm = plevm.transpose('lev','time')  # Force this ordering for constructing profiles (some cases differ)
    plevi = plevi.transpose('ilev','time')
    
    plevm.attrs['units'] = "Pa"
    plevi.attrs['units'] = "Pa"

# Height with standard atmosphere

#    print(scam_in)
    zlevm = plevm
#    print(1000.*mpc.pressure_to_height_std(plevm).values)
    zlevm_vals = 1000.*mpc.pressure_to_height_std(plevm).values
    zlevi_vals = 1000.*mpc.pressure_to_height_std(plevi).values
    dzbot = np.array(1000.*mpc.pressure_to_height_std(plevi[-1,:]).values)
   
    
#    print(plevi[-1,:])

    zlevm = plevm.copy(deep=True) # use plev variable structures on the dataset
    zlevi = plevi.copy(deep=True)
    

    zlevm.values = zlevm_vals # Just move across values to retain xarray wrapper
    zlevi.values = zlevi_vals # ""
    
    nlevm = scam_in['lev'].size # Size of vertical dimensions
    nlevi = scam_in['ilev'].size
        
# Normalize to ilev bottom being Z of surface from an expanded 2D array, then transpose back to original set up.
    
    dzbot_2d = np.tile(dzbot,(nlevm,1))  
    zlevm = np.subtract(zlevm,dzbot_2d)
    zlevm = zlevm.transpose()
    
    dzbot_2d = np.tile(dzbot,(nlevi,1))  
    zlevi = np.subtract(zlevi,dzbot_2d)
    zlevi = zlevi.transpose()
    
    
    
        
    v_coord = [plevm,zlevm] if imlev in 'mid' else [plevi,zlevi] # Return dep. on interface/mid
        
    return v_coord





######################################################
###  Derived Variable for Plotting in SCAM.        ###
######################################################	
	
def dev_vars_scam(var_name,sfile_in):
    
    plevm = sfile_in['hyam']*p0 + sfile_in['hybm']*sfile_in['PS'].isel(lat=0,lon=0)
    if 'lat' in list(plevm.dims): plevm = plevm.loc(lat=0,lon=0) # For compatability with more scripts
   
    if var_name in ['TH','THL','THV'] : 
        theta = sfile_in['T'].isel(lat=0,lon=0).transpose()*(p0/plevm)**r_cp #
        theta.attrs['long_name'] = "Potential Temperature" 
        theta.attrs['units'] = "K" 
        dev_var = theta
  
   
    if var_name=='THV' : # (For dry air)
        thetav = theta*(1.+0.61*sfile_in['Q'].isel(lat=0,lon=0).transpose())
        thetav.attrs['long_name'] = "Virtual Potential Temperature" 
        thetav.attrs['units'] = "K" 
        dev_var = thetav   
        
    if var_name =='THL': 
        thetal = dev_var-(theta/sfile_in['T'].isel(lat=0,lon=0).transpose()) \
            *(Lv/cp_air)*sfile_in['Q'].isel(lat=0,lon=0).transpose() 
        thetal.attrs['long_name'] = "Liq. Water Potential Temperature"
        dev_var = thetal
    return (dev_var)




####################################################################
###  INterpolate NCAR LES on rgular grid withe 1D coordinates    ###
####################################################################	
	
def les_reg_grid(var_les,plev,zlev):
    # Set coordinate levels of z as the mean of the z array
    print(zlev[0:20,0:150])
    print(var_les)
    
#    pvarp_sm = pvarp_sm.values.ravel()
#    plev = plev.values.ravel()
#    zlev = zlev.values.ravel()
#    hour_frac = np.repeat(hour_frac,np_les)
   
    return (var_les_reg)








#################################
### Local Time Offset from GMT
#################################


""" GET THE ZOFFSET """
""" AND FIGURE OUT THE DAYS TO GRAB """

def get_iop_info(case_iop):
  

    if case_iop == 'SAS': 
        zoffset = -6
        iop_day = -999.
        ttmin = 5. ; ttmax = 18
    if case_iop == 'PERDIGAO': 
        zoffset = 2. # CET?
        iop_day = '2017-05-23'
        ttmin = 5. ; ttmax = 20
    if case_iop == 'RICO' : 
        zoffset = -4. ##  ??
#        iop_day = '2004-12-16'
        iop_day = -999.
        ttmin = 5. ; ttmax = 20

    
    return(zoffset,iop_day,ttmin,ttmax)


#### 







####################################################################
###  OPEN AND READ SCAM OR LES IOP FILES.                        ###
####################################################################	


def scam_open_files(case_iop,file_num,run_name,dir_main):

    
# SCAM Output
 
    if file_num != 'LES' :
        
        
# File assignment using core dir from notebook
        
        files_dir = dir_main+'history/'
        
        files_pre = files_dir+'FSCAM.T42_T42.'+case_iop+'.' 
        
# NEED TO CLEAN UP THIS FILE LOGIC TO BE MORE SPECIFIC
        files_list = files_pre+file_num+'.cam.h0*00.nc'
  
#        print(files_list)

# Grab files (could be more than one)
#        print(sp.run(['ls',files_list], shell=True)) # Fix tis check to work

        try:
            stdout1 = sp.check_output('ls '+files_list, shell=True)
        except sp.CalledProcessError as e:
            print(case_iop+' - '+run_name+' - Files Not Present - EXITING...')

# Call to OS ls command

        files_in = sp.getoutput('ls '+files_list)
                
# Convert string to an array    
    
        files_in = files_in.split()
                   
# Grab some iop specific info.
            
        zoffset,iop_day,ttmin,ttmax = get_iop_info(case_iop)
    
# Check for and concatonate multiple files
# Multi
        if len(files_in) > 1:
            iop_dset = xr.open_mfdataset(files_in)
                        
        else :
            
# Single
            iop_dset = xr.open_dataset(files_in[0],engine='netcdf4') 
            
# Trim to desired data
        if iop_day != -999.: iop_dset = iop_dset.sel(time=iop_day)  # Choose day from specs. (May, 23 2017)
        
    
        iop_dset = iop_dset.load() # Load this into memory as dask does not allow indexing or something like that.  
        
  
    if file_num == 'LES' :
            
# LES Output
        les_file_dir = dir_main+'LES/'
        les_file_in = les_file_dir+run_name+'_'+case_iop+'_LES.nc' 

        
        iop_dset = xr.open_dataset(les_file_in,engine='netcdf4') 
         
       
    
    return(iop_dset)





####################################################################
### GET LES DATA FROM FILES (currently on NCAR nc file from Ned) ###
####################################################################	


def get_les_dset(les_dset,plot_df,var):

####### LES Specific #######
         

    lscale = plot_df.loc[var,'lscale']
    les_tstart = les_dset['ts'] # Start time (local?) seconds after 00
    les_time = les_dset['time'] # Time (from start?) in seconds

    les_toffset = 0. # Strange time stuff in LES?
    hour_frac_les = (les_tstart+les_time)/3600.+les_toffset  # Transform into hours
        

## Variable ##

    var_les = plot_df.loc[var,'var_les']      
    data_les = les_dset[var_les] # Read in data # 
    data_les = data_les*lscale

    plev_les = les_dset['p']
    plev = 0.01*plev_les
    zlev = les_dset['zu']

    les_pvarp = data_les

    
## Set coordinate for data
    
    nz_les = les_pvarp.sizes['nz']
    zlev = zlev[1,0:nz_les]

    
## RETURN INFO FOR. PLOTTING
    return(les_pvarp,hour_frac_les,zlev)













####################################################################
###  VARIABLE SETUPS FOR PLOTTING (cotour,ranges, etc)           ###
####################################################################	





def var_plot_setup (plot_set,all_vars_info) :

    plot_dic = {}


# 1D TIMESERIES
    if (plot_set=='1d_ts') :
    # VAR -> vscale/ymin/ymax/p_var/lscale
        plot_dic['LHFLX']  = ['Latent Heat Flux','W/m2',1.,0.,400,'wqsfc',Lv]
        plot_dic['SHFLX']  = ['Sensible Heat Flux','W/m2',1.,0., 300,'wtsfc',cp_air]
        plot_dic['TS']     = ['Surface Temperature','K',1., 290., 310.,'',1.]
        plot_dic['PBLH']   = ['Boundary Layer Depth (CAM)','meters',1., 0., 2800.,'zi_t',1.] # zi_t: height of max theta gradient
        plot_dic['PBLH_DTH'] = ['Boundary Layer Depth (dth/dz max)','meters',1., 0., 2500.,'zi_t',1.] # zi_t: height of max theta gradient
        plot_dic['PBLH_DQ'] = ['Boundary Layer Depth (dq/dz max)','meters',1., 0., 2500.,'zi_q',1.] # zi_t: height of max q gradient
        plot_dic['PBL_DQMAX'] = ['Boundary Layer dq/dz Max.','g/kg/km',1000.*1000, -100, 0.,'q',1.] # Min value of dq/dz
        plot_dic['PRECL']  = ['Large-Scale Precipitation','mm/day',86400.*1000., 0., 10.,'',1.]
        plot_dic['PRECC']  = ['Convective Precipitation','mm/day',86400.*1000., 0., 10.,'',1.]
        plot_dic['PRECT']  = ['Total Precipitation','mm/day',86400.*1000., 0., 10.,'',1.]
        plot_dic['FLNS']   = ['Surface Net Short-wave Radiation','W/m2',1., 200., 800.,'',1.]
        plot_dic['CAPE']   = ['CAPE','J/kg',1., 0., 800.,'',1.]

        
        plot_df = pd.DataFrame.from_dict(plot_dic, orient='index',
            columns=['long_name','units','vscale','ymin','ymax','var_les','lscale'])
        
        
        
        
        
        
        
# 2D TIMESERIES 
        
    if (plot_set=='2d_ts') :
        plot_dic['T'] = ['Temperature', 
        1.,260.,305.,-.5,.5,'',1.,'K']

        plot_dic['RELHUM'] = ['Relative Humidity',
            1.,10., 120.,-10.,10.,'',1.,'%']
        plot_dic['CLOUD'] = ['Cloud Fraction', 
            100., 0., 100.,-10.,10.,'',1.,'%']
        plot_dic['Q'] = ['Specific Humidity',
            1000., 1., 12.,-2,2.,'q',1000.,'g/kg']

        plot_dic['TH'] = ['Potential Temperature', \
            1., 295, 305.,-.5,.5,'t',1.,'K']
        plot_dic['THL'] = ['Liquid Water Potential Temperature', \
            1., 270, 310.,-1.,1.,'thl',1.,'K']
        plot_dic['THV'] = ['Virtual Potential Temperature', \
            1., 295, 305.,-2.,2.,'tv',1.,'K']

        plot_dic['DCQ'] = ['Humidity Tendencies - Moist Processes', \
            1000., -5., 5.,-1.,1.,'',1.,'g/kg/day']
        plot_dic['DTCOND']  = ['Temperature Tendencies - Moist Processes', \
            86400., -10., 10.,-1.,1.,'',1.,'K/day']   

        plot_dic['ZMDT'] = ['Temperature Tendencies - Deep Convection', \
            86400., -10., 10.,-1.,1.,'',1.,'K/day']
        plot_dic['ZMDQ'] = ['Humidity Tendencies - Deep Convection', \
            86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']

        plot_dic['CMFDT'] = ['Temperature Tendencies - Shallow Convection', \
            86400., -10., 10.,-10.,10.,'',1.,'K/day']
        plot_dic['CMFDQ']  = ['Humidity Tendencies - Shallow Convection', \
            86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']



    # BL/Turb. vars.                             
        plot_dic['DTV']  = ['Temperature Tendencies - Vertical Diffusion',
            86400., -15., 15.,-1.,1.,'',1.,'K/day']
        plot_dic['VD01']  = ['Humidity Tendencies - Vertical Diffusion',
            6400.*1000., -50, 50.,-1.,1.,'',1.,'g/kg/day']

        plot_dic['STEND_CLUBB']    = ['Temperature Tendencies - CLUBB', \
            86400./1000., -20, 20,-6.,6.,'',1.,'K/day'] #J/kg.s -> K/day
        plot_dic['RVMTEND_CLUBB']  = ['Humidity Tendencies - CLUBB', \
            1000.*86400, -100., 100.,-20.,20.,'',1.,'g/kg/day']

        plot_dic['WPRTP_CLUBB'] = ['w,q - Flux Covariance - CLUBB', \
            1., -0., 600.,-100.,100.,'wq_r',Lv,'W/m^2']
        plot_dic['WPTHLP_CLUBB'] = ['w,thl - Flux Covariance - CLUBB', \
            1., -100., 100.,-10.,10.,'wt_r',cp_air,'W/m^2']
        plot_dic['WPTHVP_CLUBB'] = ['w,thv - Flux Covariance - CLUBB', \
            1., -100., 100.,-5.,5.,'',1.,'W/m^2']

        plot_dic['THLP2_CLUBB'] = ['T^2 - Variance - CLUBB', \
            1., 0., 0.05,-0.01,0.01,'tt_r',1.,'K^2']   
        plot_dic['RTP2_CLUBB']  = ['q^2 - Variance - CLUBB', 1., 0., 2.5,-0.5,0.5,'qq_r',1000.*1000.,'g^2/kg^2']
    
    
    
        plot_dic['WP2_CLUBB']   = ['w^2 - Variance - CLUBB', \
            1., 0., 2.,-0.5,0.5,'ww_r',1.,'m^2/s^2'] 

        plot_dic['WP3_CLUBB']   = ['w^3 - Skewness  - CLUBSB', \
            1., 0., 0.5,-0.05,0.05,'www_r',1.,'m^3/s^3']
        
#### TURN INTO DATFRAME ####
        
        plot_df = pd.DataFrame.from_dict(plot_dic, orient='index', \
            columns=['long_name','vscale','cmin','cmax','acmin','acmax','var_les','lscale','units'])

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# 1D SNAPSHOT PROFILE PLOTS

    if (plot_set=='1d_snap') :
        
        plot_dic['T']      = ['Temperature',1.,260.,305.,-20.,20.,'',1.,'K']
        plot_dic['RELHUM'] = ['Relative Humidity',1.,10., 120.,-100.,100.,'',1.,'%']
        plot_dic['CLOUD']  = ['Cloud Fraction',100., 0., 100.,-80.,80.,'',1.,'%']
        plot_dic['Q']      = ['Specific Humidity',1000., 1., 12.,-5,5.,'q',1000.,'g/kg']
 
        plot_dic['TH']  = ['Potential Temperature',1., 295, 305.,-20.,20.,'t',1.,'K']
        plot_dic['THL']  = ['Liquid Water Potential Temperature',1., 270, 310.,-20.,20.,'thl',1.,'K']
        plot_dic['THLV']  = ['Virtual Potential Temperature', 1., 270, 310.,-20.,20.,'tv',1.,'K']
    
        plot_dic['DCQ']  = ['Humidity Tendencies - Moist Processes',1000., -5., 5.,-5.,5.,'',1.,'g/kg/day']
        plot_dic['DTCOND']  = ['Temperature Tendencies - Moist Processes',86400., -10., 10.,-10.,10.,'',1.,'K/day']   
    
        plot_dic['ZMDT']  = ['Temperature Tendencies - Deep Convection',86400., -10., 10.,-10.,10.,'',1.,'K/day']
        plot_dic['ZMDQ']  = ['Humidity Tendencies - Deep Convection',86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']
    
        plot_dic['CMFDT']  = ['Temperature Tendencies - Shallow Convection',86400., -10., 10.,-10.,10.,'',1.,'K/day']
        plot_dic['CMFDQ']  = ['Humidity Tendencies - Shallow Convection',86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']
   
    
                              
# BL/Turb. vars.                             
        plot_dic['DTV']  = ['Temperature Tendencies - Vertical Diffusion',86400., -30., 13.,-10.,10.,'',1.,'K/day']
        plot_dic['VD01']  = ['Humidity Tendencies - Vertical Diffusion',86400.*1000., -50, 50.,-8.,8.,'',1.,'g/kg/day']
    
        plot_dic['STEND_CLUBB']    = ['Temperature Tendencies - CLUBB',86400./1000., -20, 20,-2.,2.,'',1.,'K/day'] #J/kg.s -> K/day
        plot_dic['RVMTEND_CLUBB']  = ['Humidity Tendencies - CLUBB',1000.*86400, -50., 50.,-20.,20.,'',1.,'g/kg/day']
                                                       
        plot_dic['WPRTP_CLUBB'] = ['w,q - Flux Covariance - CLUBB',1., -0., 600.,-50.,50.,'wq_r',Lv,'W/m^2']
        plot_dic['WPTHLP_CLUBB'] = ['w,thl - Flux Covariance - CLUBB',1., -100., 100.,-50.,50.,'wt_r',cp_air,'W/m^2']
        plot_dic['WPTHVP_CLUBB'] = ['w,thv - Flux Covariance - CLUBB',1., -100., 100.,-50.,50.,'',1.,'W/m^2']
        
        plot_dic['THLP2_CLUBB'] = ['T^2 - Variance - CLUBB',1., 0., 0.15,-50.,50.,'tt_r',1.,'K^2']   
        plot_dic['RTP2_CLUBB'] = ['q^2 - Variance - CLUBB', 1., 0., 2.5,-0.5,0.5,'qq_r',1000.*1000.,'g^2/kg^2'] 
 
        plot_dic['WP2_CLUBB'] = ['w^2 - Variance - CLUBB', 1., 0., 1.5,-0.5,0.5,'ww_r',1.,'m^2/s^2'] 
        plot_dic['WP3_CLUBB'] = ['w^3 - Skewness  - CLUBB',1., 0., 1.5,-0.2,0.2,'www_r',1.,'m^3/s^3']
 
        plot_df = pd.DataFrame.from_dict(plot_dic, orient='index', \
            columns=['long_name','vscale','cmin','cmax','acmin','acmax','var_les','lscale','units'])

    
    
    
    
### LIST VARIABLE AND INFO IF REQUESTED ###
    
   
    if (all_vars_info):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('++++++++++++ ALL AVAILABLE VARIABLES FOR => ',plot_set,' <= ++++++++++')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        display(plot_df)
#    print(plot2d_df.style.set_table_styles([{'selector':'','props':[('border','4px solid #7a7')]}]))


    return plot_df