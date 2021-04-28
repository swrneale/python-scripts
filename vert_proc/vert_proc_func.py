######################################################
## PYTHON FUNCTIONS FOR VERTICAL PROCESSES NOTEBOOK ##
######################################################

import numpy as np
import matplotlib.pyplot as mp
import xarray as xr
import datetime as dt
import cartopy.crs as ccrs
import pandas as pd


from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)



def nino_sst_anom(sst_data,nino):

    """
        Nino/Nina SST informations
        
        Input: Timseries of SST over global region
        Output: Timseries of SST-anomalies for different nino regions.
                Classification of nino/neutral/nina based on SST anomalies

    """

    # Nino regions    
    nino_reg = {}
    
    nino_reg['nino1+2']   = [-10.,0.,270.,280.]
    nino_reg['nino3']   = [-5.,5.,210.,270.]
    nino_reg['nino3.4'] = [-5.,5.,190.,240.]   
    nino_reg['nino4']   = [-5.,5.,160.,210.]
    nino_reg['nino5']   = [-5.,5.,120.,140.]
    nino_reg['nino6']   = [8.,16.,140.,160.]
    
    print('-- Calculating for ',nino,' region')
    
    # Set nino averaging domain
    nino_s = nino_reg[nino][0] ; nino_n = nino_reg[nino][1]
    nino_w = nino_reg[nino][2] ; nino_e = nino_reg[nino][3]

    
## Be careful as thhe time a coordinate is 1 month off FEB actually = JAN as detrmined by the cftime coordinate.
    
    # Read in TS (SSTs) from h0 files  
    sst_ts = sst_data['TS'].loc[:,nino_s:nino_n,nino_w:nino_e].mean(dim=['lat','lon']) 
  
#    sst_ts = h0_month_fix(sst_ts)
   
    
    ## Remove average for each month of year (annual cycle)
      
    mnames_all = sst_data.time.dt.strftime("%b") 
    year_all = sst_data.time.dt.strftime("%Y") 
    time_axis = np.arange(0,year_all.size)
    
    # Find unique months fpor removeal of annual cycle.

    mnames = np.unique(mnames_all)
    
    # Loop of months of the year and remove the annual cycle.

    for im in mnames:
        # Match months in time series
        lmm = mnames_all==im
    
    # Detrmine indeices of matching months
        imm = [i for i, val in enumerate(lmm) if val] 
    # Average of this month
        sst_mmon = np.mean(sst_ts[imm])

    # Mopulate SST in each month to give anomaly from the annual avreage cycle.
        sst_ts = np.where(lmm,np.subtract(sst_ts,sst_mmon),sst_ts)
    
    print(time_axis)
    print(year_all)
    it_ticks = np.arange(0,len(year_all),12)
    
    fig, ax = mp.subplots(figsize=(16, 5))
    ax.plot(time_axis,sst_ts,color='white')
    ax.set_title(nino+' SSTA for '+sst_data.case)
    
    ax.set_xlabel("Year") 
    ax.set_ylabel("K") 
    ax.set_xticks(it_ticks+6)
    ax.set_xticklabels(year_all[it_ticks].values)
 
# 
    ax.fill_between(time_axis,0.,sst_ts, where=sst_ts > 0,  facecolor='red', interpolate=True)
    ax.fill_between(time_axis,sst_ts, 0., where=sst_ts < 0, facecolor='blue', interpolate=True)
    ax.xaxis.set_minor_locator(MultipleLocator(12))
    ax.tick_params(which='minor', length=7)

    
    mp.hlines(0., min(time_axis), max(time_axis), color='black',linestyle="solid",lw=1)
    mp.hlines([-np.std(sst_ts),np.std(sst_ts)], min(time_axis), max(time_axis), color='black',linestyle="dashed",lw=1)
        
    
    mp.show()    

    return sst_ts








###### CORRECT h0 FILES MONTH ########


def h0_month_fix(hist_tseries_var):
    
    year = hist_tseries_var.time.dt.year
    month = hist_tseries_var.time.dt.month
    
    print(hist_tseries_var.time.time)
    
    hist_tseries_var.time.dt.year[0] = cftime.DatetimeNoLeap(1979, 1, 1, 0, 0, 0, 0)
    
    return hist_tseries_var







##########  GET CORRECT TENDENCY VARIABLES ##########

def cam_tend_var_get(files_ptr,var_name):

# Determining CAM5/CAM6 based on levels.

    nlevs = files_ptr.lev.size
    fvers = files_ptr.variables
#

    if var_name not in ['STEND_CLUBB','RVMTEND_CLUBB']
    if var_name in fvers: print
    if nlevs in [32,30]: 
        
        
    
# Variable read in and time averaging (with special cases).

    if var_name == 'DTCOND' and : 
            var_in = files_ptr['DTCOND'].mean(dim=['time'])+files_ptr['DTV'].mean(dim=['time'])

    if var_name == 'DCQ' and case in ['rC5','rUW']: 
            var_in = files_ptr['DCQ'].mean(dim=['time'])+files_ptr['VD01'].mean(dim=['time'])
    
    if var_name == 'STEND_CLUBB':
       
            var_in = 1005.*(files_ptr['DTV'].mean(dim=['time'])
            +files_ptr['MACPDT'].mean(dim=['time'])/1000.
            +files_ptr['CMFDT'].mean(dim=['time']))
        else :
            var_in = files_ptr[var_name].mean(dim=['time'])           
    
    if var_name == 'DIV':  
            var_in = -files_ptr['OMEGA'].mean(dim=['time']).differentiate("lev")
   
    if var_name in ['OMEGA','ZMDT','ZMDQ']:
            var_in = files_ptr[var_name].mean(dim=['time'])

    return tend_var
            