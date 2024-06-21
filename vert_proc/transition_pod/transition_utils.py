import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
from cartopy.feature import LAND





'''
	Read in time-slice format data from either GFDL (AM3) or CESM (CAM6) model
	(Currently just monthly)
'''

def nino_sst_anom(model_tslice,var_tslice):
	
	dir_tslice = '/glade/campaign/cgd/amp/bundy/mdtf'
	dir_am3 = dir_tslice+'gfdl_timeslice/ts/monthly/1yr/'
	dir_cam6 = dir_tslice+'cesm_mdtfv3_timeslice_public/atm/mon/'
	
	if model_tsince ='am4' then model_suf = 
	
	file_tsice = dir_Cam6+
	
	ds_tslice = xr.open_dataset(file_in,engine='netcdf4')
	
	return ds_tslice






'''
	Calculate mean monthly climatology from monthly timeseries data
'''
