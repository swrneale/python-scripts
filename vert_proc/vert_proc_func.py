######################################################
## PYTHON FUNCTIONS FOR VERTICAL PROCESSES NOTEBOOK ##
######################################################

import numpy as np
import matplotlib.pyplot as mp
import xarray as xr
import datetime as dt
import cartopy.crs as ccrs
import pandas as pd




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
    
# Set nino averaging domain
    nino_s = nino_reg[nino][0] ; nino_n = nino_reg[nino][1]
    nino_w = nino_reg[nino][2] ; nino_e = nino_reg[nino][3]
    
# Read in TS (SSTs) from h0 files  
    sst_ts = sst_data['TS'].loc[:,nino_s:nino_n,nino_w:nino_e].mean(dim=['lat','lon']) 

    
    
    
# Remove average for each month of year (annual cycle)
  
    mnames = sst_data.time.dt.strftime("%b")[0:11] # 
    mnames_all = sst_data.time.dt.strftime("%b") 
    
    
    fig, ax = mp.subplots()
    ax.plot(sst_ts)
    mp.show()    

    return sst_ts