'''
	Utility Routines for ITCZ Analysis
'''

import xarray as xr
import matplotlib.pyplot as mp
import pandas as pd
import geocat.comp as gc
#import seaborn as sb
import scipy as sp

import numpy as np
import os
import glob as gb
import sys 




'''
	Determine if data is available (raw or derived climos) and if it needs to be saved. 
'''

def get_cesm_data(var_name,doption,dperiod,run_name,dir_in):

	print('-------- get_hist_data --------')

	# Make sure run history directories exists

	dir_path = dir_in+run_name+'/'
	climo_path = dir_path+'climo/'
	hist_stub = dir_path+'atm/hist/'+run_name+'.cam.'
	
	ldir_run = does_path_exist('Run',dir_path,'exit')
	ldir_climo = does_path_exist('Climo',climo_path,'create')
	
	
	# Create climatologies if the variable doesn't exist.
	# Document what raw data exists and whether a climo file exists.
		
	
		
	# if doption in [0,1]:
	
	if dperiod == 'm':

		climo_file = climo_path+run_name+'_clim_dmeans_ts'+var_name+'.nc.'
			
		# See if climo exits

		climo_file_exist =  os.path.isfile(climo_file)  

		# If not then try and create from infividual h1 files

		if not climo_file_exist:
		
			print('')
			print('-- MONTHLY HISTORY FILES (h0) --')

			fhist_h0 = sorted(gb.glob(hist_stub+'*h0*nc'))
			if len(fhist_h0) == 0: print('h0 files do not exist -- exiting') ; sys.exit(1)

			# First and last files in a list.	
			print('First = '+fhist_h0[0])
			print('Last  = '+fhist_h0[-1])
			
		
	if dperiod == 'd':


		climo_file = climo_path+run_name+'_clim_mmeans_ts'+var_name+'.nc.'

		# See if climo exists
		
		climo_file_exist =  os.path.isfile(climo_file)  
		
		
		if not climo_file_exist:
		
			# If not then try and create from infividual h0 files

			print('')
			print(' -- DAILY HISTORY FILES (h1) --')

			fhist_h1 = sorted(gb.glob(hist_stub+'*h1*nc'))
			if len(fhist_h1) == 0: print('h0 files do not exist -- exiting') ; sys.exit(1)


			print('First = '+fhist_h1[0])
			print('Last  = '+fhist_h1[-1])


			clivmo_var = climo_var_create(var_name,fhist_h1)
	
	
		
	
	lreturn = True
	
	return lreturn
	
	
	
'''
	Construct Climatologies from history input data (daily or monthly)
'''
	
def climo_var_create(var_name,hist_files):

	print('')		
	print('-- Opening History Files --')
	dset_hist = xr.open_mfdataset(hist_files,parallel=True)


	# Is the variable avalailable?
	dset_vars = list(dset_hist.keys())
	print(dset_vars)
	
	if var_name not in dset_vars : print(var_name+' not in history files- exiting') ; sys.exit(1)
	print(dset_hist[var_name])


	# Check for complete years






	lreturn = True

	return lreturn

	
def get_var_details():
	lhold=True

	
	
	
	
'''
	Path Checks for Model and CLimatology File Data
'''
	
def does_path_exist(pname,path_in,path_action):
	
	lpath_exist = True if  os.path.exists(path_in) else False
	
	print(path_in+' -- '+pname+' path exists') if lpath_exist else print(path_in+' -- '+pname.upper()+' PATH DOES NOT EXIST')	   
	
	if not lpath_exist and path_action == 'create': os.mkdir(path_in)
	if not lpath_exist and path_action == 'exit'  : print('+++++++++ EXITING +++++++++') ; sys.exit(0)
	
	return
	
	
		   
		   


