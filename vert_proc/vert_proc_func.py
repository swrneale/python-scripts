######################################################
## PYTHON FUNCTIONS FOR VERTICAL PROCESSES NOTEBOOK ##
######################################################

import numpy as np
import matplotlib.pyplot as mp
import xarray as xr
import datetime as dt
#import monthdelta 
from dateutil.relativedelta import relativedelta
import cartopy.crs as ccrs
import pandas as pd
import pygrib as pyg # Read in grib for analyses (ECMWF)



from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
















"""
        Nino/Nina SST information
        
        Input: Timseries of SST over global region
        Output: Timseries of SST-anomalies for different nino regions.
                Classification of nino/neutral/nina based on SST anomalies

"""


def nino_sst_anom(run_case,sst_data,nino):

   

    # Nino regions (S/N/W/E) 
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

    
## Be careful as the time a coordinate is 1 month off FEB actually = JAN as detrmined by the cftime coordinate.
    
# Read in TS (SSTs) from inputdata HadISST for now.  
#    print(sst_data)
    sst_ts = sst_data.loc[:,nino_s:nino_n,nino_w:nino_e].mean(dim=['lat','lon']) 
    sst_ts = sst_ts.compute()
   
#    sst_ts = h0_month_fix(sst_ts)
   

## IF TS/SST data comes from history files may need to check that first file should be Jan and Not Feb    
## Remove average for each month of year (annual cycle)
#    sst_data.time = sst_data.time + dt.timedelta(month=1)
#    sst_sata.time = sst_data.time + relativedelta(-1)

    
    mnames_all = sst_data.time.dt.strftime("%b") 
    year_all = sst_data.time.dt.strftime("%Y") 
    time_axis = np.arange(0,year_all.size)
    print(mnames_all)
    
    # Find unique months for removal of annual cycle.

    mnames = np.unique(mnames_all)
    
    # Loop of months of the year and remove the annual cycle.

    ''' FIND AND REMOVE ANNUAL CYCLES '''
    print(mnames)
    print(mnames_all)
  
    print(year_all)

    

    ### REMOVE ANNUAL CYCLE ##

#    sst_ts = sst_ts.resample('M')


    for imname in mnames :
        print(imname)
        imon_ts = mnames_all == imname
        sstm = sst_ts[imon_ts].mean()
        sst_ts[imon_ts] = sst_ts[imon_ts] - sstm

  
    
#    for im in mnames:
    # Match months in time series (logical).
#        lmm = mnames_all==im
#        print(lmm)
        
    # Determine indices of matching months
#        imm = [i for i, val in enumerate(lmm) if val] 
#        print(imm)
        
    # Average of this month
#        sst_mmon = np.mean(sst_ts[imm])
#        print(sst_ts[imm])
#        print(sst_mmon)

    # Populate SST in each month to give anomaly from the annual average cycle.
#        sst_ts = np.where(lmm,np.subtract(sst_ts,sst_mmon),sst_ts)
 #   print(sst_ts)
   
    ''' PLOTTING '''

    
    it_ticks = np.arange(0,len(year_all),12)
    
    fig, ax = mp.subplots(figsize=(16, 5))
    
    
    print(sst_ts.time)

    ax.plot(time_axis,sst_ts,color='black')
#    ax.set_title(nino+' SSTA for '+run_case)
    
    print("hi2")
    
#    ax.set_xlabel("Year") 
#    ax.set_ylabel("K") 
#    ax.set_xticks(it_ticks+6)
#    ax.set_xticklabels(year_all[it_ticks].values)
 
    print("hi3")
    ax.fill_between(year_all,0.,sst_ts, where=sst_ts > 0,  facecolor='red', interpolate=True)
    ax.fill_between(year_all,sst_ts, 0., where=sst_ts < 0, facecolor='blue', interpolate=True)
    ax.xaxis.set_minor_locator(MultipleLocator(12))
    ax.tick_params(which='minor', length=7)

    print("hi4")
#    mp.hlines(0., min(sst_ts.time), max(time_axis), color='black',linestyle="solid",lw=1)
#    mp.hlines([-np.std(sst_ts),np.std(sst_ts)], min(time_axis), max(time_axis), color='black',linestyle="dashed",lw=1)
        
    
    mp.show()    

    return nino_ts,nina_ts











'''
#################################### 
###### CORRECT h0 FILES MONTH ######
#################################### 
'''

def h0_month_fix(hist_tseries_var):
    
    year = hist_tseries_var.time.dt.year
    month = hist_tseries_var.time.dt.month
    
    print(hist_tseries_var.time.time)
    
    hist_tseries_var.time.dt.year[0] = cftime.DatetimeNoLeap(1979, 1, 1, 0, 0, 0, 0)
    
    return hist_tseries_var







'''
#####################################################
##########  GET CORRECT TENDENCY VARIABLES ##########
#####################################################
'''


def cam_tend_var_get(files_ptr,var_name):

# Determining CAM5/CAM6 based on levels.

    nlevs = files_ptr.lev.size
    fvers = files_ptr.variables
#

#    if var_name not in ['STEND_CLUBB','RVMTEND_CLUBB']
#    if var_name in fvers: print
#    if nlevs in [32,30]: 
        
        
    print(np.any(np.isin(files_ptr.variables,var_name)))
# Variable read in and time averaging (with special cases).

#    if var_name == 'DTCOND' and : 
#            var_in = files_ptr['DTCOND'].mean(dim=['time'])+files_ptr['DTV'].mean(dim=['time'])

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

    return var_in
            
    
    
'''
#########################################################
    COMMON ROUTINE FOR SETTING UP FILES (CAM/Analyses)
#########################################################
'''

## Should get month means of analyses, CAM (h0 and ts files)    
## > ERA5 CISL-RDA ds633.1 : /glade/collections/rda/data/ds633.1/e5.moda.an.pl
## - 1 netcdf file/year (1979=2018)
## > MERRA2  CISL-RDA ds313.3 : /glade/collections/rda/data/ds613.3/1.9x2.5/
## Res. files (not clear where monthly means are) - GRIB!!!   
## >ERA-interim CISL-RDA ds627.1 : /glade/collections/rda/data/
##
## >JRA-55 CISL-RDA ds628.9 : /glade/collections/rda/data/ds628.9/
##
    
def get_files_type(case_name,case_type,var_cam,years) :

    
    type_desc = {}
    type_desc['cam'] = ['/glade/p/rneale']


    allowed_types = ['cam','reanal']

    if case_type not in allowed_types : print(case_type+ ' files - type not allowed')
    if case_type     in allowed_types : print(case_type+ ' files - type allowed') 

    print('-Grabbing data type/case -- '+case_type+' '+case_name)
 

    yr0 = years[0]
    yr1 = years[1]

## GRAB ANALYSIS ##

    lat_rev = False
    lcoord_names = False

    
    if var_cam != 'TS':
        
        if case_type=='reanal' :
            dir_rda = '/glade/collections/rda/data/'
            if case_name=='ERA5' :
                var_anal_fmap = {'T': 't',   'Q':'q'}
                var_anal_vmap = {'T': 'T',   'Q':'Q'}
                var_vname = var_anal_vmap[var_cam] ; var_fname = var_anal_fmap[var_cam] 
                rda_cat = 'ds633.1'

                dir_glade = dir_rda+rda_cat+'/'
                files_glade  = np.array([dir_rda+rda_cat+"/e5.moda.an.pl/%03d/e5.moda.an.pl.128_130_%s.ll025sc.%03d010100_%03d120100.nc"%(y,var_fname,y,y) for y in range(yr0,yr1+1)])
                print(files_glade)
                lat_rev = True
                lcoord_names = True
            
            if case_name=='ERAI' :
                var_anal_fmap = {'T': 't',   'Q':'q'}
                var_anal_vmap = {'T': 'T',   'Q':'Q'}
                var_vname = var_anal_vmap[var_cam] ; var_fname = var_anal_fmap[var_cam] 
                if var_cam in ['T'] : var_fname = 'sc'
                if var_cam in ['U','V'] : var_fname = 'uv' 
                rda_cat = 'ds627.1'

                dir_glade = dir_rda+rda_cat+'/'
                files_glade  = np.array([dir_rda+rda_cat+"/ei.moda.an.pl/ei.moda.an.pl.regn128%s.%03d%02d0100.nc"%(var_fname,y,m) for y in range(yr0,yr1+1) for m in range(1,12)])
                print(files_glade)
                print('hi4')
            
            
            if case_name=='MERRA2' : #### NOT CLEAR MMEAN DATA AVAILABLE
                resn = '1.9x2.5'
#            var_anal_fmap = {'T': '',   'Q':'q'}
                var_anal_vmap = {'T': 'T',   'Q':'Q'}
                var_vname = var_anal_vmap[var_cam] 
                rda_cat = 'ds613.3'

                dir_glade = dir_rda+rda_cat+'/'
                files_glade  = np.array([dir_rda+rda_cat+"/%s/%03d/MERRA2%03d010100_%03d120100.nc"%(resn,y,y,y) for y in range(yr0,yr1+1)])
                print(files_glade)
            
    
            
#### GRAB CAM SST AMIP DATASET FOR NOW FOR ANALYSES
        
    if (var_cam=='TS') :
        print('- Grabbing file(s) for AMIP and REANALYSES from CESM inputdata -')
        dir_inputdata = '/glade/p/cesmdata/cseg/inputdata/atm/cam/sst/'
        hadisst_file = 'sst_HadOIBl_bc_0.9x1.25_1850_2020_c210521.nc'
        files_glade = dir_inputdata+hadisst_file
        var_vname = 'SST_cpl'

    
            
            
## POINT TO FILES ##

    
    data_files = xr.open_mfdataset(files_glade,parallel=True,chunks={"time": 1})
    
#    data_files = xr.open_mfdataset(files_glade)
    
    
## STANDARDIZE COORDS/DIMS ##
    
    if lcoord_names : data_files = data_files.rename({'latitude':'lat', 'longitude':'lon', 'level':'lev'})
    
# Reverse lat array to get S->N if needed
    if lat_rev : data_files = data_files.reindex(lat=list(reversed(data_files.lat)))
#    print(data_files)
    
        
    return data_files,var_vname








''' INPUT DATASET DESCRIPTIONS '''

def cam_reanal_list():
	

    rl = {} # Revert List
# BL Vres
    rl['ERA5']   =  ['ERA5']
    rl['ERAI']   =  ['ERAI']



# Data frame
    rl_df = pd.DataFrame.from_dict(rl, orient='index',columns=['run name'])
    return rl_df