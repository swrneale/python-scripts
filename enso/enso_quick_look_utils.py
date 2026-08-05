'''
    Utility Routines for ENSO Diagnostics (currently a redo of the ncl quick look panels)
'''

import xarray as xr
import numpy as np
import pandas as pd

import glob as glob
import os as os



### Mya need to CHANGE for your own local directory to write out derived timeseries (not used for LENS, already in tseries format.)
#dir_ncout = '/glade/work/rneale/python-netcdf/enso/'
dir_ncout = '/glade/derecho/scratch/rneale/enso_wavelet/'
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



def get_dataset(case,case_type,case_owner,var_axis,yr0,yr1,lread_in_all_hist,lwrite_ts_file ,lread_ts_file):


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


                case 'HADISST':

                    dir_hadisst = '/glade/campaign/cesm/cesmdata/cseg/inputdata/atm/cam/sst/sst_HadOIBl_bc_0.9x1.25_1850_2022_c241003.nc'
        
                    da_axis = xr.open_dataset(dir_hadisst)['SST_cpl']
                    vscale = 1.
            
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



        case _ if case_type in ['cesm1','cesm2','cmip6']:

            vscale = cvar_scale
            
            if case_type == 'cesm1':
                dir_lens = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/'
                fyrs_str = '.04*'
                file_suff = var_axis+'/'+case+'.cam.h0.'+var_axis+fyrs_str+'.nc'
            
            if case_type == 'cesm2':
                dir_lens = '/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/month_1/'
                fyrs_str = '.18*'
                file_suff = var_axis+'/'+case+'.cam.h0.'+var_axis+fyrs_str+'.nc'

            if case_type == 'cmip6':    
                dir_lens = '/glade/campaign/collections/cdg/data/CMIP6/CMIP/NCAR/CESM2/piControl/r1i1p1f1/Amon/ts/gn/v20190320/'
                file_suff = 'ts_Amon_CESM2_piControl_r1i1p1f1_gn_040001-049912.nc'
                var_axis = 'ts'

               
                
         
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
            # Check for this file name lower
            # Do logic to pick up either h0 or h0a filename

            ofile_case =  dir_ncout+case+'.cam.h0a.TS.'+str(yr0)+'-'+str(yr1)+'.10.N-10.S.nc'
            
      
            # Add vscale here
            
            vscale = cvar_scale

           
            
            if (os.path.exists(ofile_case) or os.path.exists(ofile_case.replace('h0a','h0'))) and lread_ts_file:

             
                
                if os.path.exists(ofile_case.replace('h0a','h0')): ofile_case = ofile_case.replace('h0a','h0')

                
                
                print('  - Timeseries files exist for the case - so I am using them')
            
                print('     ',ofile_case)
            
                da_axis = xr.open_dataset(ofile_case)[var_axis]

               
              

            else:
                
                # Pick the right directory (hannay orgmarques)
    
                dir_c3 = '/glade/derecho/scratch/'+case_owner+'/archive/'
                
#                if os.path.isdir(dir_c3+case):
#                    print("   - Cecile's Run")
#                # Your operation here, e.g. read/write files
#                else:
#                    print("   - Gustavo's Run")
#                    dir_c3 = '/glade/derecho/scratch/gmarques/archive/'
    
                
    # Grab files and read in.
    
                # Trim down range of files to read in requested years, otherwise read in all.
    
                print(lread_in_all_hist)
                if not lread_in_all_hist:
                    print('  - Using a subset of the available data')
                    files_hist = []
                    yrange = list(range(yr0, yr1+1))
    #                yr_arr_string = "[" + ",".join(f"{n:04d}" for n in yrange) + "]"
                    yr_arr_strings = [f"{num:04d}" for num in yrange]
    
                    for yr_str in yr_arr_strings:
                        file_ls = dir_c3+case+'/atm/hist/'+case+'.cam.h0a.'+yr_str+'*.nc'
                        print(file_ls)
                        files_hist.extend(glob.glob(file_ls)) 
    
                    files_hist.sort()
        
                
                else:
                    print('  - Using all of the available data')
                    print(case)
                    files_ls = dir_c3+case+'/atm/hist/'+case+'.cam.h0a.*.nc'
                    files_hist = glob.glob(files_ls)
                    
                files_hist.sort()

     
     # File checks           
                
                print('  - Reading ',len(files_hist),' files (first/last)')
                print('   -',files_hist[0])
                print('   -',files_hist[-1])
                
            # Open them as multiple files
    
                print('  - Opening detected h0a output files...')
                da_axis = xr.open_mfdataset(files_hist,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",chunks={})[var_axis]
                
                print('  -Done')
                
            # Write out the timeseries of 2D field files (unscaled)?
            
                if lwrite_ts_file: 
    
                    print("  - Write out files of 2D field "+var_axis)
        
                    print('    ',ofile_case)
        
        
                    da_axis.to_netcdf(ofile_case,mode="w")
                    da_axis.close()

    
                    print(' - Done')

        case _ if case_type in ['e3smv1','e3smv2','e3smv3']:

              print('  - Grabbing file(s) for E3SMv1/v2 ensembles '+case)

              vscale = 1.
              dir_ccr  = '/glade/campaign/cgd/ccr/'
            
              if case_type == 'e3smv1': dir_e0 = dir_ccr+'E3SMv1-LE/FV_regridded/'  ; tsuff = '.cam.h0.TS.000101-050012'
              if case_type == 'e3smv2': dir_e0 = dir_ccr+'E3SMv2/FV_regridded/'     ; tsuff = '.eam.h0.TS.0040101-050012'
              if case_type == 'e3smv3': dir_e0 = dir_ccr+'E3SMv3-LE/'               ; tsuff = '.en00.TS.000101-050012'


              dir_e = dir_e0+case+'/atm/proc/tseries/month_1/'
              efile_case = dir_e+case+tsuff+'.nc'

              da_axis = xr.open_dataset(efile_case)[var_axis]
    
    
        case _:
            
            print('  - No case_type match for '+case)


# Just scale the variable right at the end

  
# Different data formats require different ways to slice the years.
    if type(da_axis.time.values[0]) == 'cftime._cftime.DatetimeNoLeap':
        da_axis = da_axis.sel(time=slice(str(yr0), str(yr1)))
    else:
        da_axis = da_axis.sel(time=(da_axis.time.dt.year >= yr0) & (da_axis.time.dt.year <= yr1))
        
    da_axis = vscale *  da_axis
    
    return da_axis

















''' Calculate El Nino Anomalies timeseries'''


def nino_anom_ts(da_axis,nino_reg):

    from scipy import signal

    # nino and var regions
    # # Taking settings from the Clivar 2020 ENSO metrics
    
    
    nino_w,nino_e,nino_n,nino_s = nino_region(nino_reg)
    
    
    nino_axis = da_axis.sel(lat=slice(nino_s, nino_n), lon=slice(nino_w, nino_e))
    
    
    # Step 3: Area-weighted average over lat/lon
    weights = np.cos(np.deg2rad(nino_axis.lat))
    
    nino_waxis = nino_axis.weighted(weights).mean(dim=["lat", "lon"])
    
    
    # Group by month, calculate climatology (mean for each calendar month)
    nino_caxis = nino_waxis.groupby('time.month').mean('time')
    
    
    # Subtract monthly climatology to get anomalies
    
    nino_anom = nino_waxis.groupby('time.month') - nino_caxis


    
    # Linearly detrend the data only
    nino_anom.values = signal.detrend(nino_anom.values)
   
    
    # Convert to NumPy and flatten, mask NaNs
#    nino_1d = nino_anom.values.flatten()
    
#    nino_1d = nino_1d[~np.isnan(nino_1d)] 
    
    
    
    
    
    return nino_anom









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


'''
    CALCULATION AND PLOTTING FUNCTIONS
'''


def calc_power(ax,nino_ts):

    from scipy.signal import welch
    from statsmodels.tsa.ar_model import AutoReg
    from scipy.stats import chi2
    
    # Example input: nino_reg is your time series (numpy array)
    # Replace with your own data
#    nino_reg = np.random.randn(600)  # e.g., 50 years of monthly anomalies

    nino_ts = nino_ts.dropna("time").values  
#    display(nino_ts)

    # Parameters

    jave = 5     # number of segments to average (Welch method)
    pct = 0.10   # percent taper (not directly used, approximate with window)
    
    
    # --- Compute power spectrum using Welch ---
    nperseg = len(nino_ts) // jave
    freqs, psd = welch(nino_ts, window="hann", nperseg=nperseg, scaling="density")
    
    # --- Convert frequency to period in years (assuming monthly data) ---
    periods = 1 / (freqs * 12)
    
    # --- Estimate AR(1) red noise ---
    model = AutoReg(nino_ts, lags=1, old_names=False).fit()
    phi = model.params[1]   # lag-1 autocorrelation
    var = np.var(nino_ts)
    
    red_noise = (1 - phi**2) / (1 - 2*phi*np.cos(2*np.pi*freqs) + phi**2) * var
    
    # --- Confidence intervals (chi-square) ---
    dof = 2 * jave  # degrees of freedom ~ 2*number of averages
    alpha_low, alpha_high = 0.05, 0.95
    
    
    lower = dof / chi2.ppf(1-alpha_low/2, dof)
    upper = dof / chi2.ppf(alpha_low/2, dof)
    
    ci_low = red_noise * lower
    ci_high = red_noise * upper
    
    # --- Plot ---

    ax.plot(periods, psd, color="black", lw=2, label="Spectrum")
    ax.plot(periods, red_noise, color="red", lw=2, label="Red noise")
    ax.plot(periods, ci_low, "r--", lw=1, label="5% / 95% CI")
    ax.plot(periods, ci_high, "r--", lw=1)
    
    ax.set_xlim(8, 0.0833)  # mimic NCL reversed axis
    ax.set_ylim(0, 40)
    
    ax.set_xlabel("Period (years)", fontsize=12)
    ax.set_ylabel("Variance (unit² / freq)", fontsize=12)
    ax.set_title("Power Spectrum", fontsize=14)
    
#    ax.legend()
    ax.invert_xaxis()  # Reverse x-axis like in NCL
    xp = [1,2,3,4,5,6,7,8]
    ax.set_xticks(xp)
    ax.set_xticklabels(xp)




'''
    SCALE THE EXTENT OF THE DATA ON THE X-AXIS
'''


def set_fractional_xlim(ax_list, xdata, frac=0.7, anchor="left"):
    """
    Expand xlim so that the data only fills a fraction of the axis width.
    Hides tick labels outside the actual data range.
    """
    
    x = np.asarray(xdata)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    D = xmax - xmin
    if not (0 < frac <= 1):
        raise ValueError("`frac` must be in (0, 1].")
    W = D / frac  # total axis width so data spans `frac` of it

    if anchor == "left":
        xl, xr = xmin, xmin + W      # start exactly at xmin
    elif anchor == "right":
        xl, xr = xmax - W, xmax      # end exactly at xmax
    elif anchor == "center":
        mid = 0.5 * (xmin + xmax)
        xl, xr = mid - 0.5 * W, mid + 0.5 * W
    else:
        raise ValueError("ENSO_QL: anchor must be 'left', 'center', or 'right'")

    for ax in ax_list:
        ax.set_xlim(xl, xr)

        # relabel xticks: blank them if outside data range
        ticks = ax.get_xticks()
        labels = []
        for t in ticks:
            if xmin <= t <= xmax:
                labels.append(f"{t:g}")
            else:
                labels.append("")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    ax.margins(x=0)

    return xl, xr





