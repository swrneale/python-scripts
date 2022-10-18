import numpy as np
import xarray as xr
import pandas as pd

from metpy.interpolate import log_interpolate_1d
from metpy.units import units

import matplotlib.pyplot as mp
from matplotlib import cm
#from mpl_toolkits.axes_grid1 import AxesGrid
#from mpl_toolkits.basemap import Basemap as bmap
import matplotlib as mpl

import seaborn as sbn



obs_names = ['TRMM','GPCP','NVAP','ERAI','MERRA']


'''
	Routines for bar chart plots over the MC
'''

def read_climo (cname,vname,season):
		
	print('-- Reading climo data in and constucting/reading land mask --')	
	
	odir_in = '/glade/p/cesm/amwg/amwg_data/obs_data/'
	cam_idir = '/glade/p/cesmdata/cseg/inputdata/atm/cam/topo/'
	
	mdir_in = '/glade/scratch/rneale/archive/'
#	cedir_in = '/glade/scratch/cmip6/archive/'
	
	
	
	dcode_times=True
	
	
	if cname in obs_names:
		
		file_in = odir_in+cname+'_'+season+'_climo.nc' 
		dcode_times = False
	
	else:
				
		file_in = mdir_in+cname+'/atm/climo/'+cname+'_'+season+'_climo.nc'
	

	print('  - Reading in: '+file_in)
	dset_in = xr.open_dataset(file_in,engine='netcdf4',decode_times=dcode_times)
	

# Need to generate, interpolate and add LANDMASK if not available.	

	if "LANDFRAC" not in dset_in.variables or cname in ('f.e22.FHIST.f09_f09.cesm2_2.mcont_mjo_ocn.001','f.e22.FHIST.f09_f09.cesm2_2.mcont_mjo_ocn_c2sst.001'):
		
		lfrac_in = cam_idir+'USGS_gtopo30_0.23x0.31_remap_c061107.nc'
		dset_lfrac = xr.open_dataset(lfrac_in,engine='netcdf4')
		dset_lfrac = dset_lfrac.interp_like(dset_in)
		dset_in['LANDFRAC'] = dset_lfrac['LANDFRAC']
				
			
			
	print('-- Done --')
			
	return dset_in 



def reg_mask(ds_in,cname,vname,var_mask_reg,land_frac_thresh):
	
	print('-- Masking variable data --')
	
#	min_lat = -12. ; max_lat=12.
#	min_lon = 90. ; max_lon = 150.
	
	min_lat = -18. ; max_lat=18.
	min_lon = 90. ; max_lon = 160.
	
# Grab land fraction
	
	lfrac = ds_in.LANDFRAC
	
	var_in = get_var(ds_in,vname)
	
	lfrac_mc = lfrac.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
	var_in_mc = var_in.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
	
# Variable meta
	

#	var_meta = get_var_meta(vname)

	var_units = get_var_meta().loc[vname]['units'] 

	
# Populate datafrane
	
	var_mask = np.zeros(3)
	mask_names = ['Ocean','Land','Both']
	
	
	var_mask[0] = var_in_mc.where(lfrac_mc <= land_frac_thresh[0]).mean()
	var_mask[1] = var_in_mc.where(lfrac_mc >= land_frac_thresh[1]).mean()
	var_mask[2] = var_in_mc.mean()

	var_mask_reg_case = pd.DataFrame({'Case':cname,'Mask':mask_names[i],var_units:var_mask[i]} for i in range(3))
	
	var_mask_reg = pd.concat([var_mask_reg,var_mask_reg_case],ignore_index=True)
	
	
	return var_mask_reg
	
	
	
	




def get_var(ds,varname):
	
	print('-- Grabbing Data According to requested variable')

	
	if varname=='PRECT':
		if 'PRECT' not in ds.variables:
			vscale = 86400.*1000.
			var_in = vscale*(ds['PRECL']+ds['PRECC'])
		else:
			var_in = ds.PRECT
		
	else:

		
		if varname not in ds.variables:
			
			vname = get_var_meta().loc[varname]['obs_var']  
			print(vname)
			var_in = ds[vname]
			
			if var_in.ndim != 3:
				var_in = var_in.sel(lev=500.)
			
		else:
		
			try:
				var_in = ds[varname]
			except:
				print(varname+' not found...')

	print('-- Done --')	
			
	return var_in




def reg_bar_plot(bplot,var_masked):
	
	
	sns.barplot(data=var_masked_reg)
	
	
	return bplot



 
def get_var_meta():


	
	# List of variable info.

	pvars = {}

	pvars['PRECT']  = ['','Precipitation','mm/day','2d','x','x',
		86400.*1000,0.,15.,-8.,8.,'terrain_r','PRGn']
	pvars['LHFLX']  = ['','Surface Latent Heat Flux','W/m^2','2d','x','x',
		1,50.,150.,-140.,140.,'PuBuGn','PRGn']
	pvars['SHFLX']  = ['','Surface Sensible Heat Flux','W/m^2','2d','x','x',
		1,50.,150.,-100.,100.,'PuBuGn','PRGn']
	pvars['SWCF']   = ['','Short Wave Cloud Forcing','W/m^2','2d','x','x',
		1.,-100,-10.,-50.,50.,'Purples_r','RdBu_r']
	pvars['LWCF']   = ['','Long Wave Cloud Forcing','W/m^2','2d','x','x',
		1.,10.,85.,-30.,30.,'Reds','RdBu_r']
	pvars['CLDTOT'] = ['','Total Cloud','%','2d','x','x',
		100.,0,100.,-30.,30.,'PuBuGn','PRGn']
	pvars['CLDLOW'] = ['','Total Cloud','%','2d','x','x',
		100.,0,100.,-30.,30.,'PuBuGn','PRGn']
	pvars['TMQ']  = ['PREH2O','Precipitable Water','mm','2d','x','x',
		1,40.,60.,-8.,8.,'terrain_r','PRGn']
	
	pvars['OMEGA500']  = ['OMEGA','Precipitable Water','mm','2d','x','x',
		1,40.,60.,-8.,8.,'terrain_r','PRGn']

	# Data frame
	pvars_df = pd.DataFrame.from_dict(pvars, orient='index',columns=['obs_var','Long Name','units','dim_var','3dvar','plevel','vscale','vmin','vmax','avmin','avmax','cols','acols'])
	
	
	return pvars_df
