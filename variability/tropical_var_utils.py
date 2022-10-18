'''
        TROPICAL VARIABILITY UTILITIES
'''


import xarray as xr
import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt

import os 
import sys
import glob as gb
import datetime as dt

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import cm
from matplotlib.dates import DayLocator, HourLocator, MonthLocator, DateFormatter

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cf

#from distributed import Client
#from ncar_jobqueue import NCARCluster

from scipy.signal import butter, lfilter, lfilter_zi, freqz

from IPython.display import clear_output
import time as timing







'''
        Convert Model Variable to Observed Variable
'''


def mvar2ovar(model_vname,case_name):
        
	var_name = None
       
	if (case_name in ['TRMM','NOAA']):
        
		if ((case_name == 'TRMM') & (model_vname =='PRECT')) : var_name = 'precip' 
		if ((case_name == 'NOAA') & (model_vname =='FLUT'))  : var_name = 'FLUT' 
		if ((case_name == 'HadISST') & (model_vname =='TS'))  : var_name = '' 
                
		if var_name == None : print('Observed data source and var name are inconsistant') ; sys.exit()
                
                
	else:

		var_name = model_vname
		
		if case_name in ['CGCM','AGCM_5dRandPatt','AGCM_1dRandPt','AGCM_1dRandPatt']: 
			var_name = 'TS'+'_'+case_name
			if 'anom' in model_vname: var_name = 'TS'+'_'+case_name+'_anom'
			
		if case_name == 'CGCM_bg'   : var_name = 'TS'+'_'+case_name
		if case_name == 'AGCM_mon'  : var_name = 'SST_cpl'
		
		
	return var_name
	


'''
        Variability Data Routines
'''


def trop_data_daily(dir_in,run_name,set_name):
        
       
	print('-- trop_data_daily --')
         
	if run_name=='TRMM': 
                
# Obs.
	
		files_list = '/glade/p/cgd/amp/rneale/data/TRMM/daily/3B42.1998-2013.daily.1deg.v7.nc'
		ds_run = xr.open_dataset(files_list, decode_cf=True, decode_times=True) 
                
	elif run_name=='NOAA':
# Obs.
		files_list = '/glade/work/rneale/data/NOAA/NOAA_dmeans_ts_FLUT.nc' 
		ds_run = xr.open_dataset(files_list, decode_cf=True, decode_times=True) 
                
# Need to reverse latitude for NOAA.
		ds_run = ds_run.reindex(lat=ds_run.lat[::-1])
               
        
	elif run_name  in ['AGCM_mon','CGCM','CGCM_bg','AGCM_5dRandPatt','AGCM_1dRandPt','AGCM_1dRandPatt']:
		
		if run_name == 'CGCM_bg':
			files_list = dir_in+set_name[1]+'.'+set_name[0]+'.TS.LatBlend_30.CGCM.1980-2014.nc'
		elif run_name == 'CGCM_mon':
			files_list = '/glade/p/cesmdata/cseg/inputdata/atm/cam/sst/sstice_b.e21.BHIST.f09_g17.CMIP6-historical.011_0.9x1.25_yrs1850-2014_timeseries_c20190404.nc'
			
		else:
			files_list = dir_in+set_name[1]+'.'+set_name[0]+'.TS.LatBlend_30.'+run_name+'.1980-2014.nc'	
		
#		files_list = dir_in+'sstice_b.e21.BHIST.f09_g17.CMIP6-historical.011_0.9x1.25_yrs1980-2014_daily_timeseries_c20191114_'+run_name+'.nc'
		
		
		ds_run = xr.open_dataset(files_list, decode_cf=True, decode_times=False) 
                
# Need to reverse latitude for NOAA.
                
        
	else :
        
# Model
        
		files_star = dir_in+run_name+'/atm/hist/*h1*.nc'
		files_list = sorted(gb.glob(files_star))
       
		print(' -> # Files/First/last names...')
		print(len(files_list))
		print(files_list[0])
		print(files_list[-1])

		ds_run = xr.open_mfdataset(files_list, decode_cf=True, decode_times=True, parallel=True, chunks={"time": 365}) 

# Make sure it is datetime64 so it can be plotted nicely with datetime objects.
#	ds_run['time'] = ds_run.indexes['time'].to_datetimeindex()

	
	units, reference_date = ds_run.time.attrs['units'].split('since')
		
	if run_name != 'CGCM_mon':
		ds_run ['time'] = pd.date_range(start=reference_date, periods=ds_run.sizes['time'], freq='D')
	else: 
		ds_run ['time'] = pd.date_range(start=reference_date, periods=ds_run.sizes['time'], freq='M')
	
	ds_run = ds_run.transpose("time", "lat", "lon")
	
	

        
	return ds_run


        

# Read in a variable
def trop_data_getvar(da_run,var_name,year_range,lat_range,lon_range):
        
	print('-- trop_data_getvar --')
                
	year_first,year_last = year_range
	lats,latn = lat_range
	lonw,lone = lon_range
	
	run_var = da_run[var_name].sel(time=slice(str(year_first),str(year_last)))
	
	run_var = run_var.sel(lon=slice(lonw,lone))
#	run_var.sel(time="1982-07").plot(figsize=(30,30))
#	plt.show()
	run_var = run_var.sel(lat=slice(lats,latn))		
#	run_var = run_var.mean(dim='lat')

# Make sure it is datetime64 so it can be plotted nicely with datetime objects.
#	run_var = run_var.astype("datetime64[ns]") 

       
	
        
	return run_var














'''
        Variability Processing Routines
'''






def trop_data_filt(data_in,ftype):
   
	import matplotlib.pyplot as plt
	from scipy.signal import freqz

	print('-- trop_data_filt --')     



# Sample rate and desired cutoff frequencies (in Hz).
	fs = 1. # samples per day
	window = 200
	cutoff_low = 1./100. # Low pass cuttoff (slow - days)
	cutoff_high = 1./20. # High pass cutoff (fast - days)



	data_filt = lanczos_band_pass(data_in, window, cutoff_low, cutoff_high, dim='time')



	return data_filt
        
        
   







'''
        Variability Plotting
'''

def trop_data_plot(axn,run_name,data_var,years_plot):
	
	print('-- trop_data_plot --')

	if 'anom' in data_var.name:
		print('-- Anomalies--')
#		clevels = [-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80]	 # FLUT anoamlies (OLR)
		clevels = [ii*0.5 for ii in [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0]]
		cmap_sst = 'RdYlBu_r'
	else:
		print('-- Full Field --')
		clevels = [25,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32]
		cmap_sst = 'CMRmap_r'
	


	year_start,year_end = [str(x) for x in years_plot]
#	lonw,lone = lon_range

    
# Subset data in time and remove mean.
#	data_plot = data_var.sel(lat=5., method="nearest").sel(time=slice(year_start,year_end))

#	data_var = data_var-data_var.mean(dim='time')

	

	pplot = axn.contourf(data_var.lon,data_var.time,data_var,levels=clevels,cmap=cmap_sst)
	pplot1 = axn.contour(data_var.lon,data_var.time,data_var,levels=clevels)

#	axn.set_xticks(np.arange(0, 360+1.,20.))   
	lon_formatter = LongitudeFormatter()
	axn.xaxis.set_major_formatter(lon_formatter)
	

	axn.yaxis.set_major_locator(MonthLocator())
#	axn.yaxis.set_minor_locator(MonthLocator())
	axn.yaxis.set_major_formatter(DateFormatter('%b %Y'))
#	fign.autofmt_xdate()
			
	axn.set_title(run_name)
	
	plt.rcParams.update({'font.size': 22})
	axn.grid(True,linewidth=2)
	

	return axn,pplot



#class trop_data:

#''' Using Variability Data Class '''
#    

#  def __init__(self, length, breadth, unit_cost=0):
#       self.length = length
#       self.breadth = breadth
#       self.unit_cost = unit_cost
#   
#   def data_read(self,files_in):
#        
#        files_list = sorted(gb.glob(files_in))
#        da_run = xr.open_mfdataset(files_list, decode_cf=True, decode_times = True, parallel=True) 
#        print(da_run['PRECT'])
#        
#        return da_run 
#   

# breadth = 120 cm, length = 160 cm, 1 cm^2 = Rs 2000
#r = Rectangle(160, 120, 2000)

#print("Area of Rectangle: %s cm^2" % (r.get_area()))
#print("Cost of rectangular field: Rs. %s " %(r.calculate_cost()))






	





        




""" 
		Low, High and BandPass filtering
"""







def lanczos_low_pass_weights(window, cutoff):
    """
    Calculate weights for a low pass Lanczos filter.

    Inputs:
    ================
    window: int
        The length of the filter window (odd number).

    cutoff: float
        The cutoff frequency(1/cut off time steps)

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
#     sigma = 1.   # edit for testing to match with Charlotte ncl code
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def lanczos_low_pass(da_ts, window, cutoff, dim='time', opt='symm'):
    
    wgts = lanczos_low_pass_weights(window, cutoff)
    weight = xr.DataArray(wgts, dims=['window_dim'])
    
    if opt == 'symm':
        # create symmetric front 
        da_ts = da_ts.transpose('lat','lon','time')
        da_front = (xr.DataArray(da_ts.loc[
                                    dict(time=slice("%0.4i-01-01"%da_ts['time.year'][0],
                                                    "%0.4i-12-31"%da_ts['time.year'][0]))].values,
                                dims=['lat','lon','time'],
                                coords=dict(lat=da_ts.lat.values,
                                            lon=da_ts.lon.values,
                                            time=da_ts.loc[
                                                dict(time=slice("%0.4i-01-01"%da_ts['time.year'][0],
                                                                "%0.4i-12-31"%da_ts['time.year'][0]))].time.values
                                                                -dt.timedelta(days=365)))
                   )
        da_front = da_front.reindex(time=list(reversed(da_front.time.values)))
        
        # create symmetric end
        da_end = (xr.DataArray(da_ts.loc[
                                  dict(time=slice("%0.4i-01-01"%da_ts['time.year'][-1],
                                                  "%0.4i-12-31"%da_ts['time.year'][-1]))].values,
                                dims=['lat','lon','time'],
                                coords=dict(lat=da_ts.lat.values,lon=da_ts.lon.values,
                                            time=da_ts.loc[
                                                dict(time=slice("%0.4i-01-01"%da_ts['time.year'][-1],
                                                                "%0.4i-12-31"%da_ts['time.year'][-1]))].time.values
                                                                +dt.timedelta(days=365)))
                 )
        da_end = da_end.reindex(time=list(reversed(da_end.time.values)))
        
        da_symm = xr.concat([da_front,da_ts,da_end],dim='time')
        da_symm_filtered = da_symm.rolling({dim:window}, center=True).construct('window_dim').dot(weight)
        da_ts_filtered = da_symm_filtered.sel(time=da_ts.time)
        
    else:
        da_ts_filtered = da_ts.rolling({dim:window}, center=True).construct('window_dim').dot(weight)
    
    return da_ts_filtered
    
def lanczos_high_pass(da_ts, window, cutoff, dim='time'):
    
    da_ts_lowpass = lanczos_low_pass(da_ts, window, cutoff, dim='time')
    da_ts_filtered = da_ts-da_ts_lowpass
    
    return da_ts_filtered    

def lanczos_band_pass(da_ts, window, cutoff_low, cutoff_high, dim='time'):
    
    da_ts_filtered = lanczos_low_pass(da_ts, window, cutoff_high, dim='time')
    da_ts_filtered = lanczos_high_pass(da_ts_filtered, window, cutoff_low, dim='time')
    
    return da_ts_filtered
























        
''' SCIPY: Generate Butter Bandpass filter '''

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass',analog=False)
              
              
              
              
              

              
    return b, a





''' SCIPY: Apply generated filter to the input data (3D) '''

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
# Construct filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        
### Plot the frequency response for a few different orders.

    plt.figure(1)
    plt.clf()
    for order in [3, 4, 5,6]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.15 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.xlim([0,0.15])
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
        

        
# Loop for 1D routines :(
    ilons = range(len(data.lon)) ; ilats =  range(len(data.lat))
    print(b,a)
#    data_out = [lfilter(b, a, data[:,iy,ix]) for iy in ilats for ix in ilons]

    it0 = 0
    it1 = 1200
    lat_loc = 0.
    lon_loc = 150
      
   

    start = timing.process_time()

# Vectorize ??

#    filterl_vec = np.vectorize(lfilter,excluded=['a','b'])

    zi = lfilter_zi(b, a)

    data_filt = data.copy()      
    data_filt.values = lfilter(b,a,data,axis=zi*data[0])          

                         
#    for ilon in ilons[0:40]:
#        for ilat in ilats[0:40]:
#            data_out = lfilter(b, a, data[:,ilat,ilon]) 
#            clear_output(wait=True)
#            print(ilon,ilat)
        
    print('time filter = ',timing.process_time() - start)
   
        
    plt.figure(figsize=(12, 6))

   
    dplot = data.sel(lon=lon_loc,lat=lat_loc, method="nearest")
    dplot_filt = data_filt.sel(lon=lon_loc,lat=lat_loc, method="nearest")


   
    dplot_filt = dplot_filt+np.average(dplot)
        
    plt.plot(dplot.values,color='blue')
    plt.plot(dplot_filt,color='red')
    plt.show()

    

 
        

        
    return data_filt










