'''
    Utility Routines for ENSO Diagnostics (currently a redo of the ncl quick look panels)
'''

import xarray as xr
import numpy as np
import pandas as pd

import glob as glob
import os as os



### Mya need to CHANGE for your own local directory to write out derived timeseries (not used for LENS, already in tseries format.)
dir_ncout = '/glade/work/rneale/python-netcdf/enso/'
dir_data = '/glade/work/rneale/data/'



''' Set nino123456 domain '''

def nino_region(nino_name):

# Same lat range for now.cesm1
    nino_s = -5. ;  nino_n = 5.

    match (nino_name):
        case 'nino3':  nino_w = 210 ;  nino_e = 270.
        case 'nino34': nino_w = 190 ;  nino_e = 240.
        case 'nino4':  nino_w = 160 ;  nino_e = 210.
            


    return nino_w,nino_e,nino_n,nino_s







''' Read in Different Datasets For each Axis '''



def get_dataset(case,case_type,var_axis,yr0,yr1,lread_in_all_hist,lwrite_ts_file ,lread_ts_file):


    cvars = ['TS','TAUX','PRECT','OMEGA500','DTCOND300','DTCOND500','DTCOND700'] 
    cvar_scales = [1.,1.,86400.*1000.,36.,86400.,86400.,86400.]
    cvar_scale = cvar_scales[cvars.index(var_axis)]

# Obs. variable names.
    evars = ['sst','chnk','tp','w','']
    efvars = ['sst','taux','prect','omega500','']
    
    match case:




    
        case _ if case_type == 'OBS':


            # Selct obs. source.

            print('  - Grabbing ',case,' data for',var_axis)
            
            match case:

                case 'ERA5':

                    
                    
                    dir_era5 = '/glade/derecho/scratch/rneale/ERA5/mmean/1deg/'
                  
       
# Some ERA5 variable mappings to CESM vars.

                    evar = evars[cvars.index(var_axis)]
                    efvar = efvars[cvars.index(var_axis)]
                    
                    da_axis = xr.open_dataset(dir_era5+efvar+'_era5_monthly_1x1.nc')[evar]
                    
                    if 'valid_time' in da_axis.dims:
                        da_axis = da_axis.rename({'valid_time': 'time'})

                    vscale = 1.
                    
                    if var_axis == 'TS':    vscale = 1.
                    if var_axis == 'PRECT': vscale = 1000.
                    if var_axis == 'TAUX':  vscale = 30. 
                    if var_axis == 'OMEGA500':  vscale = 36.     

                case 'TROPFLUX' if var_axis=='TAUX':
                    
                    da_axis = xr.open_dataset(dir_data+'tropflux/taux_tropflux_1m_1979-2018.nc')['taux']    
                    da_axis = da_axis.rename({'latitude': 'lat', 'longitude': 'lon'})
                    vscale = -1.

                case _:
                    
                    print("  - No obs, product match for ",case_type)



        case _ if case_type in ['cesm1','cesm2']:

            vscale = cvar_scale
            
            if case_type == 'cesm1':
                dir_lens = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/'
                fyrs_str = '.04*'
            else:
                dir_lens = '/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/month_1/'
                fyrs_str = '.18*'
                
            file_suff = var_axis+'/'+case+'.cam.h0.'+var_axis+fyrs_str+'.nc'
            files_hist = dir_lens+file_suff
            files_ls  = glob.glob(files_hist)         

      
            
            
            print('  - Grabbing file(s) for LENS1/2 (CESM1/2)')
           
#            da_axis = xr.open_mfdataset(files_ls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")[var_axis]
            
            
            if case_type=='cesm1' and var_axis == 'PRECT': # Need to grab PRECC and PRECL files separately 
    
                files_hist_pc = files_hist.replace('PRECT', 'PRECC')
                files_hist_pl = files_hist.replace('PRECT', 'PRECL')
    
                files_ls_pc  = glob.glob(files_hist_pc)
                files_ls_pl  = glob.glob(files_hist_pl)
                
                da_pc = xr.open_mfdataset(files_ls_pc,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")['PRECC']
                da_pl = xr.open_mfdataset(files_ls_pl,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")['PRECL']
                
                da_axis = da_pc+da_pl
                
            else:

                # If empty file then try my derived directory
                if not files_ls:
                    print('-Checking for local CESM copy, likely a derived variable if it exists')
                    files_hist = dir_ncout+file_suff
                    files_ls  = glob.glob(files_hist)
    
                
                da_axis = xr.open_mfdataset(files_ls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")[var_axis]
              
                
        
           
    
        case _ if case_type == 'cesm3':


            # Does a timeseries dataset exist for this case?
            # Check for this file name lwoer down    

            
            ofile_case = dir_ncout+case+'_'+var_axis+'_mmeans_ts.nc'

      
            # add vscale here
            vscale = cvar_scale
            
            if os.path.exists(ofile_case) and lread_ts_file:
    
                print('  - Timeseries files exist for the case - using them')
            
                print('     ',ofile_case)
            
                da_axis = xr.open_dataset(ofile_case)[var_axis]


            else:
                
                # Pick the right directory (hannay/gmarques)
    
                dir_c3 = '/glade/derecho/scratch/hannay/archive/'
                
                if os.path.isdir(dir_c3+case):
                    print("   - Cecile's Run")
                # Your operation here, e.g. read/write files
                else:
                    print("   - Gustavo's Run")
                    dir_c3 = '/glade/derecho/scratch/gmarques/archive/'
    
                
    # Grab files and read in.
    
                # Trim down range of files to read in requested years, otherwise read in all.
    
                
                if not lread_in_all_hist:
                    files_hist = []
                    yrange = list(range(yr0, yr1+1))
    #                yr_arr_string = "[" + ",".join(f"{n:04d}" for n in yrange) + "]"
                    yr_arr_strings = [f"{num:04d}" for num in yrange]
    
    #                print(y_arr_strings)
                    for yr_str in yr_arr_strings:
    #                    print(yr_str)
                        file_ls = dir_c3+case+'/atm/hist/'+case+'.cam.h0a.'+yr_str+'*.nc'
                        files_hist.extend(glob.glob(file_ls)) 
    
                    files_hist.sort()
                
                else:
                    file_hist = dir_c3+case+'/atm/hist/'+case+'.cam.h0a.*.nc'
                    
               
     
     # File checks           
                
                print('  - Reading ',len(files_hist),' files  (first/last)')
                print('   -',files_hist[0])
                print('   -',files_hist[-1])
                
            # Open them as multiple files
    
                print('  - Slow read of h0a output...')
                da_axis = xr.open_mfdataset(files_hist,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")[var_axis]
                
    
                print('  -Done')
                
            # Write out the timeseries of 2D field files (unscaled)?
            
                if lwrite_ts_file: 
    
                    print("  - Write out files of 2D field "+var_axis)
        
                    print('    ',ofile_case)
        
        
                    da_axis.to_netcdf(ofile_case,mode="w")
                    da_axis.close()

    
                    print(' - Done')
    
        case _:
            
            print('  - No case_type match for '+case)


# Just scale the variable right at the end


    da_axis = vscale *  da_axis

    

    return da_axis












''' Check Requested Time Overlap and Trim to a Common Period '''


def dataset_ts_trim(da_x,da_y,yr0,yr1,var_x,var_y,get_months):


# Time details for x axis

    print('  - Requested variable year range = ',yr0,'-',yr1)
    
    print('  - Time details for x-axis - '+var_x)
    fyr0 = da_x.time.min().dt.strftime('%Y.%b').item()
    fyr1 = da_x.time.max().dt.strftime('%Y.%b').item()
    print('    - Available year range = ',fyr0,'-',fyr1)

    print('  - Time details for y-axis - '+var_y)
    fyr0 = da_y.time.min().dt.strftime('%Y.%b').item()
    fyr1 = da_y.time.max().dt.strftime('%Y.%b').item()

    da_x_min = da_x.time.min().values
    da_x_max = da_x.time.min().values
    da_y_min = da_y.time.min().values
    da_y_max = da_y.time.min().values
    
    print('    - Available year range = ',fyr0,'-',fyr1)
    

    if (da_x_min != da_x_max and da_y_min != da_y_max):
        print('    - DataArray time range of axes do not match')


# Subset time range (years, seasons) from available full range. Used padded 4 digit integers.


    tx_mask = ((da_x.time.dt.year >= yr0) & (da_x.time.dt.year <= yr1) &
                    (da_x.time.dt.month.isin(get_months)))
    
    da_x = da_x.sel(time=tx_mask)

    
    ty_mask = ((da_y.time.dt.year >= yr0) & (da_y.time.dt.year <= yr1) &
                    (da_y.time.dt.month.isin(get_months)))

    da_y = da_y.sel(time=ty_mask)

    
# Check they have the same time dimenstion noe.

    

    
#    print(da_y.time.values)
#    x0, x1 = pd.to_datetime(da_x.time.values[0]),pd.to_datetime(da_x.time.values[-1])
#    y0, y1 = pd.to_datetime(da_y.time.values[0]),pd.to_datetime(da_y.time.values[-1])

#    span_x = x1.year - x0.year + 1
#    span_y = y1.year - y0.year + 1

 #   print('     - Subsetted data ime span is ',[span_y == span_x])
    
       
    return da_x, da_y











''' Calculate El Nino Anomalies timeseries'''


def nino_anom_ts(da_axis,nino_reg,axis_vals):

    from scipy.stats import gaussian_kde

    
# nino and var regions
## Taking settings from the Clivar 2020 ENSO metrics


    nino_w,nino_e,nino_n,nino_s = nino_region(nino_reg)

    
    nino_axis = da_axis.sel(lat=slice(nino_s, nino_n), lon=slice(nino_w, nino_e))
    
    
    # Step 3: Area-weighted average over lat/lon
    weights = np.cos(np.deg2rad(nino_axis.lat))
    
    nino_waxis = nino_axis.weighted(weights).mean(dim=["lat", "lon"])
    
    
    # Group by month, calculate climatology (mean for each calendar month)
    nino_caxis = nino_waxis.groupby('time.month').mean('time')

    
    # Subtract monthly climatology to get anomalies
    
    nino_anom = nino_waxis.groupby('time.month') - nino_caxis

    
    # Convert to NumPy and flatten, mask NaNs
    
    nino_1d = nino_anom.values.flatten()

    nino_1d = nino_1d[~np.isnan(nino_1d)] 


    
    # Strip out requested season before constructing PDFs

    ''' PDFs of monthly variable '''

    nino_kde = gaussian_kde(nino_1d, bw_method=0.3)

    nino_pdf = nino_kde(axis_vals)

  
    

    return nino_1d, nino_pdf









'''
    PLOTTING FUNCTIONS
'''





''' Set some plot domain ranges for an axis '''

def fig_domains(vname):

    match vname:
        case 'TS':
            vmin,vmax = -3.,3.
        case 'PRECT':
            vmin,vmax = -4.,4.
        case 'TAUX':
            vmin,vmax = -0.08, 0.08
        case 'OMEGA500':
            vmin,vmax = -0.8, 0.8   
        case _ if vname in ['DTCOND500','DTCOND300','DTCOND700']:
            vmin,vmax = -3, 3   
           
        case _:
            
             print('  - Variable not recognized '+vname)
            
# Retun array where gaussian_kde will fit to.
    
    axis_vals = np.linspace(vmin, vmax, 100)

    return vmin,vmax,axis_vals






    