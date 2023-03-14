
''' UTILITES FOR MONSOON MEAN PLOTS FOR CATALYST PAPERS  '''

import glob
import xarray as xr
import pandas as pd




'''
	Months of year short names 
'''

def msnames():
	msnames_out = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	return msnames_out


''''''
''' Calculate Monthly Phase of Maximum and Minimum '''
''''''

def calc_mon_phase(var_climo):


## Calculate the phase min and max of climo. data at each grid point.

	mnames = var_climo.time.dt.strftime("%b") 

	phase_min = var_climo.argmin(dim='time').values
	phase_max = var_climo.argmax(dim='time').values

	climo_phase_max= xr.DataArray(phase_max, coords=[var_climo.lat, var_climo.lon], dims=["lat", "lon"])
	climo_phase_min= xr.DataArray(phase_min, coords=[var_climo.lat, var_climo.lon], dims=["lat", "lon"])


	return climo_phase_min,climo_phase_max




''''''
''' Grab and/or construct monthly climos '''
''''''

def get_mon_climos(case,case_df,var,var_df,years,l_ens_ave):

	

	lexist = case_df.loc[case,'lcexist'] 
	run_name = case_df.loc[case,'rname'] 
	case_dir = case_df.loc[case,'cdir']                        
	lobs = case_df.loc[case,'lobs']
	fin_type =  case_df.loc[case,'cclimo']

	vscale = var_df.loc[var,'voscale'] if lobs else var_df.loc[var,'vscale'] 





# Obs. always assume same structure and that they exist.
	if lobs:

		print('-Observations case')

		files_in = case_dir+case+'_??_climo.nc'

		flist = sorted(glob.glob(files_in))

		print(' -> First/last files = \n '+ \
			  flist[0]+' \n '+flist[-1])

		ds_case = xr.open_mfdataset(flist,engine='netcdf4', decode_times = False, combine = 'nested') 
		ds_case['time'] = pd.to_datetime(pd.date_range("2000", freq="M", periods=12))

		var_climo = ds_case[var]



#### Grab existing full or var specific climo file

	else :

		print('-Model case')

# Do the existing model climos exist?

		if (lexist):

			print('-Full Climo files exist')

			case_dir = '/glade/scratch/cmip6/archive/'+run_name+'/atm/proc/climo/'+run_name+'/'+run_name+'.'+years[0]+'-'+years[1]+'/'

			files_in = case_dir+run_name+'.cam.h0.'+years[0]+'-'+years[1]+'._??_climo.nc'

			flist = sorted(glob.glob(files_in))


			print(' -> First/last files = \n '+ \
			  flist[0]+' \n '+flist[-1])

#            ds_case = xr.open_mfdataset(flist,engine='netcdf4', decode_times = False, combine = 'nested') 
			ds_case = xr.open_mfdataset(flist,engine='netcdf4')
			ds_case['time'] = pd.to_datetime(pd.date_range("2000", freq="M", periods=12))

			var_climo = ds_case['PRECC']+ds_case['PRECL']

			del(files_in,flist)

		else:

			'''
			 Are they in tseries format?  
			'''

			if fin_type == 'tseries':

				print('-Time series files')

				if case == 'CESM2':

					if l_ens_ave:
						files_in = case_dir+'/'+run_name+'.'+var+'.208101-210012.nc'
					else:
						files_in = case_dir+var+'/'+run_name+'.cam.h0.'+var+'*.nc'



					flist = sorted(glob.glob(files_in))
				
					ds_case = xr.open_mfdataset(flist,engine='netcdf4') 

					ds_case = ds_case.sel(time=slice(years[0],years[1]))
					var_in = ds_case[var]



				if case == 'E3SMv2':
						
						
					if var=='PRECT':
					
						if l_ens_ave:
							files_in = case_dir+'/'+run_name+'.eam.h0.PRECC.208101-210012.nc'
							lfiles_in = case_dir+'/'+run_name+'.eam.h0.PRECL.208101-210012.nc'
						else:
							files_in = case_dir+'/'+run_name+'/atm/proc/tseries/month_1/'+run_name+'.eam.h0.PRECC.nc'
							lfiles_in = case_dir+'/'+run_name+'/atm/proc/tseries/month_1/'+run_name+'.eam.h0.PRECL*.nc'

						flist = sorted(glob.glob(files_in))
						lflist = sorted(glob.glob(lfiles_in))

						print(files_in)
						
						ds_case = xr.open_mfdataset(files_in,engine='netcdf4') 
						lds_case = xr.open_mfdataset(lfiles_in,engine='netcdf4') 

						## Subset for required years.
						ds_case = ds_case.sel(time=slice(years[0],years[1]))
						var_in = ds_case['PRECC']

						lds_case = lds_case.sel(time=slice(years[0],years[1]))
						lvar_in = lds_case['PRECL']

						var_in += lvar_in


					else:

						if l_ens_ave:
							files_in = case_dir+'/'+run_name+'.eam.h0.'+var+'*.nc'
						else:
							files_in = case_dir+'/'+run_name+'/atm/proc/tseries/month_1/'+run_name+'.eam.h0.'+var+'*.nc'
						print(files_in)
						flist = sorted(glob.glob(files_in))

						ds_case = xr.open_mfdataset(files_in,engine='netcdf4') 

						ds_case = ds_case.sel(time=slice(years[0],years[1]))
						var_in = ds_case[var]

				print('-> First/last files = \n '+ \
					flist[0]+' \n '+flist[-1])






## name of years and months      
				
				mnames_all = ds_case.time.dt.strftime("%b") 
				year_all = ds_case.time.dt.strftime("%Y") 

## Initialize climo array
				var_climo = var_in[0:12,:,:]
				var_climo['time'] = pd.to_datetime(pd.date_range("2000-01", freq="M", periods=12))



## Loop imonths and month names
				for im,imname in enumerate(msnames()) :    ## Construct monthly climos.
					imon_ts = mnames_all == imname
					var_climo[im-1,:,:] = var_in[imon_ts,:,:].mean(dim='time')

#                var_climo=var_in.groupby('time.monthofyear').mean()


				'''            
					Or h0 history format?
				'''            
			else: 

## Write file out.
				print('-> Writing climo files out')



		var_climo = vscale*var_climo



	return var_climo














