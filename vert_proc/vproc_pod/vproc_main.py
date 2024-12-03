'''
    Program plots profiles of state variables and process tendencies at various locations and times of ENSO phase
    Level 1: Mean profiles of states and tendencies during ENSO phase (seasons: monthly means)
    Level 2: Time varying profiles during a season or seasonal transition
    Level 3: Statistical reltiosnhips between vertical processes and ENSO/forcing/dynamical strength
    Level 4: 
'''




import numpy as np
import matplotlib.pyplot as mp
import xarray as xr
import datetime as dt
#from dateutil.relativedelta import relativedelta


import cartopy.crs as ccrs
import pandas as pd
import dask as ds

import sys
import warnings

warnings.filterwarnings("ignore", message="FutureWarning")


# To Import My Functions ###
import vproc_func as mypy
import vproc_figs as myfigs
import vproc_case_desc as mycases
import vproc_setup as mysetup

import importlib


#from distributed import Client

''' Bring in function routines '''

importlib.reload(mypy) # Required because I am constantly the .py files
importlib.reload(mycases) 
importlib.reload(myfigs) 
importlib.reload(mysetup) 



def main():

    #	client = Client(cluster)
    #	client
    
    ''''' Which case(s) to use '''''
    
    case_desc,case_type,reanal_climo,pref_out = mysetup.vprof_setup()
    
    ''''' Which nino SST region '''''
    nino_region = 'nino34'
    
    
    
    
    ''' SEASON '''
    
    seas_mons = np.array(["Jan","Feb","Dec"])
    
    clim_anal = True
    
    ''''' Years for the analysis '''''
    
    years_data = (1979,2005) # Year range of history files to read AND either 'climo' one file or 'tseries' many files
    
    
    ''' REGIONAL SPECS (LAT/LON/LEV) '''
    
    lats_in = -45,45
    lonw_in = 0. ; lone_in = 360.
    ppmin = 50. ; ppmax = 1050.
    
    
    
    ''''' Variable description (varlot_scat is the 2nd variable used in scatter plots) '''''
    
    var_plot = 'DIV'
    var_plot_scat = 'OMEGA'
    
    ldiv = False  # Calculate divergence from OMEGA if var_cam = OMEGA
    l_psst_nino = False
    l_pminmax_plev = False # Plot lat lon plot of climo/nino/nina ma/min levels of occurrence.
    l_pscatt_2d = True # Scatter plot of 2 2D fields.
    l_vprof = False
    
    ''''' Named Regions '''''
    
    reg_df = mysetup.vprof_set_regions()
    myfigs.vprof_reg_plot(reg_df)
    
    
    
    
    
    ''''' Directory Information '''''
    
    #	dir_croot = '/glade/p/cgd/amp/people/hannay/amwg/climo/' # Directories with climo files
    #	dir_hroot = '/glade/p/cgd/amp/amwg/runs/' # Run firectories with history files
    
    dir_proot = '/glade/u/home/rneale/python/python-figs/vert_proc/'
    #	dir_obs = '/glade/p/cesm/amwg/amwg_data/obs_data/'
    
    
    ''''' Variable Meta Info. '''''
    
    var_df = mysetup.vprof_set_vars()
    
    
    
    
    
    # Pressure range info.
    
    p_levs = np.arange(ppmin,ppmax,50.)
    
    
    # Map simulation names to case names
    
    sim_names = mycases.mdtf_case_list()
    
    
    
    display(reg_df)
    display(var_df)
    
    #	reg = list(reg_names.keys())[0]
    
    reg = reg_df.index.values.tolist()[0]
    
    print(reg)
    
    nmnths = seas_mons.size
    ncases = case_desc.size
    nregions = reg_df.index.size    
    
    yr0 = years_data[0]
    yr1 = years_data[1]
    

    
    
    
    
    '''
    ########################
    ##### LOOP CASES  ######
    ########################
    '''

    ''' Set Dictonary For Plotting profiles '''
    vproc_cases = {}
    
    
    for icase,case in enumerate(case_desc): # Do first so don't have to do a read mutliple times
    
    
        
        # Grab run name 
    
        sim_name = sim_names.loc[case]['run name']
    
        lclimo = True if reanal_climo and case_type[icase] == 'reanal' else False
    
        print('')
        print('')
        print('')
        print('**** **** **** **** **** **** **** **** **** ')
        print('**** CASE # ',icase+1,' OF ',ncases,' ****')
        print('**** **** **** **** **** **** **** **** **** ')
        print('- Name = ',case,' ->',sim_name)
        print('**** **** **** **** **** **** **** **** **** ')
        print('')   
    
    
        ## Read data in from files ##
    
    
        print('-- SET TIME RANGE OF TIMESERIES DATA -- ',yr0,' to ',yr1)
        print('')
        print('-- Grabbing variable files --')
    
        if lclimo:  # Read in tseries based files here for the analysis variable
            files_ptr,var_read   = mypy.get_files_climo(sim_name,case_type[icase],var_plot,lats_in,p_levs,years_data) # Grab variable
        else :
            files_ptr,var_read   = mypy.get_files_tseries(sim_name,case_type[icase],var_plot,lats_in,p_levs,years_data) # Grab variable
            
    
    
    
        ## TS FROM HISTORY FILES (just copy for h0 files if they are already read in)
        ## Can still do this for lclimo as it will take observed if reanal
    
        print('-- Grabbing Sea Surface Temperature (SST) files --')
    
        if case_type[icase] in ['cam6_revert']: # I think this effectively acts as a pointer, I hope!
            tfiles_ptr = files_ptr 
            tvar_read = 'TS'
        else :   
            tfiles_ptr,tvar_read = mypy.get_files_tseries(sim_name,case_type[icase],'TS',lats_in,p_levs,years_data) # Grab TS for nino timeseries
    
    
        # Grabbing PS if needed
    
        print('-- Grabbing Surface Pressure (PS) files --')
    
        if case_type[icase] in ['cam6_revert','cesm3_dev']: # Grab the LENS time series or just use existing files_ptr from h0 type output.
            pfiles_ptr = files_ptr 
        else:
            if not lclimo: # Don't need to read in PS for climos.
                pfiles_ptr,pvar_read = mypy.get_files_tseries(sim_name,case_type[icase],'PS',lats_in,p_levs,years_data) # Grab TS for nino timeseries
            else :
                pfiles_ptr=None
    
    
    
        ''' TRIM FOR SPECIFIED YEARS '''
    
    
        print('-- Calculating and plotting nino SST anomalies - this will never be climo currently')
    
        sst_data = tfiles_ptr[tvar_read]
    
    
    
        ''' SST ANOMALY ROUTINE ARRAY '''
    
        sst_months =  sst_data.time.dt.strftime("%b")    
        inino_mons,inina_mons = mypy.nino_sst_anom(sim_name,sst_data,nino_region,l_psst_nino)
    
        print('-- NINO grab:  Done --')
        
        ''' 
        #############################################################
        ### FORK FOR CLIMO VERSUS h0/TSERIES INPUT FILE FORMAT ?
        #############################################################   
        '''     
    
        varp_in_lev,var_in_ps = mypy.derive_nino_vars(lclimo,var_read,var_plot,p_levs,files_ptr,pfiles_ptr,case_type[icase],var_df,inino_mons,inina_mons,seas_mons)
    
    
    
        '''
        ###############################################################
        ### Plot values of maximum of 2 fields against each other   ###
        ###############################################################
        '''     
    
        if l_pminmax_plev:
    
            print('-- Plotting max/min pressure level of field --')
            
            pdiv_lev = myfigs.plot_div_pres(case_type[icase],case,var_plot,varp_in_lev,var_in_ps,files_ptr)
    
    
    
        '''
            ####################################    
            ### Plot Scatter of 2 Quantities
            ####################################
        '''     
        
        if l_pscatt_2d:
            
            print('-- Scatter Plots of ... ---')
            
    # Need to grab second quantity
        
            if lclimo:  # Read in tseries based files here for the analysis variable
                files_ptr,var2_read   = mypy.get_files_climo(sim_name,case_type[icase],var_plot_scat,lats_in,p_levs,years_data) # Grab variable
            else :
                files_ptr,var2_read   = mypy.get_files_tseries(sim_name,case_type[icase],var_plot_scat,lats_in,p_levs,years_data) # Grab variable
            
                
            varp2_in_lev,var_in_ps = mypy.derive_nino_vars(lclimo,var2_read,var_plot_scat,p_levs,files_ptr,pfiles_ptr,case_type[icase],var_df,inino_mons,inina_mons,seas_mons)
            
            myfigs.scat_plot(case_type[icase],case,var_plot,var_plot_scat,varp_in_lev,varp2_in_lev,var_in_ps,reg_df,files_ptr)
    
    
    
        '''
            ################################################    
            ### Now Loop Regions For Vertical Profiles ###
            ################################################ 
        ''' 
        
        for ireg,reg in enumerate(reg_df.index):  ## 4 regions let's assume ##
         
        
        ### Assign lat/lon region domain ###
    
           

            '''
                ### Set region info and subset data ###
            '''
            
            varp_tavs,reg_name,reg_s,reg_n,reg_w,reg_e = mypy.vprof_set_region(ireg,reg_df,varp_in_lev)
            
    
            '''
            ########################################################################   
            ### Add to the plotting dictionary climo/nino/nina periods ###
            ########################################################################
            '''     
    
 
            
            print(' -- Adding data -- Climo/Nino/Nina Period = ') 
            
            ''' Collate data to be plotted (tricky as vertical levels can vary, set up a dictionary then) ''' 
            ''' So this is climo, nino,nina response for this region '''
          
            if ireg == 0 : # First entry for this case
                vproc_case = varp_tavs
                
            else: # Else append the array to the key entry
                vproc_case = xr.concat([vproc_case, varp_tavs],'column')

        ''' Add to dictionary for each case '''           
        vproc_cases[case] = vproc_case

   
    
    # Plot regions all together,
    if l_vprof:
        myfigs.vprof_clim_nino(vproc_cases,p_levs,var_plot,reg_df,var_df,case_type,case_desc,years_data,pref_out)                

    # 
   
    print()
    print()
    print('-- End --')
    
    
    
    
    
    
    
    
