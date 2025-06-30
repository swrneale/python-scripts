'''
    Utility Routines for ENSO Diagnostics (currently a redo of the ncl quick look panels)
'''

import xarray as xr
import numpy as np



dir_h0 = 
dir_ts = '/glade/derecho/scratch/rneale/enso_wavelet/'


''' Get SST data (obs or model'''

def enso_sst_get(cases,ystart,yend,nino='3.4',dtrend=Ture):

    ncases = cases.size



    ds_case = {}
    
    for icase,case in enumerate(cases):
        
        match(ens_name):
                
            case if case in ['HadISST','NOAA-OI']:
                files_in = 
                ds_run = xr.open_mfdataset(files_in,parallel=True,chunks=chunk_sizes)
            case if 'b.e13' in case:
                dir0 = '/glade/derecho/scratch/hannay/archive/'
                
        # Append datasets
    
            if icase==0 :
                ds_case = ds_this_case
            else:
                ds_case = ds_case+ds_this_case        
    
    
    return nino_sst





''' Set nino123456 domain '''

def nino_region(nino_name):

# Same lat range for now.
    nino_s = -5. ;  nino_n = 5.

    match (nino_name):
        case 'nino3':  nino_w = 210 ;  nino_e = 270.
        case 'nino34': nino_w = 190 ;  nino_e = 240.
        case 'nino4':  nino_w = 160 ;  nino_e = 210.
            


    return nino_w,nino_e,nino_n,nino_s







''' Read in Different Datasets '''



def get_dataset(case_type,var_x,var_y):


    cvars = ['TS','TAUX','PRECT'] 

    match case
    
       case 'ERA5':

            dir_era5 = '/glade/derecho/scratch/rneale/ERA5/mmean/1deg/'
            evars = ['sst','chnk','tp']
    
# Some variable mappings
            evar_x = evars(cvars.index(var_x))
            evar_y = evars(cvars.index(var_y))

            da_x = xr.open_dataset(dir_era5+var_x+'_era5_monthly_1x1.nc')[evar_x]
            
            if is_in_situ:
                da_y = xr.open_dataset(dir_data+'tropflux/taux_tropflux_1m_1979-2018.nc')['taux']    
                da_y = da_var.rename({'latitude': 'lat', 'longitude': 'lon'})
                vscale = 1.e3
            else:
                da_y = xr.open_dataset(dir_era5+var_y+'_era5_monthly_1x1.nc')[evar_y]  
                vscale = ovar_scale
            
            if 'valid_time' in da_y.dims:
                da_y = da_y.rename({'valid_time': 'time'})
    
            


        case _ if ctype in ['cesm1','cesm2']:
            
            if ctype == 'cesm1':
                dir_lens = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/'
                files_hist_x = dir_lens+cvar_x+'/'+cname+'.cam.h0.'+cvar_x+'.04*.nc'
                files_hist_y = dir_lens+cvar_y+'/'+cname+'.cam.h0.'+cvar_y+'.04*.nc'
                files_ls_x  = glob.glob(files_hist_x)
                files_ls_y  = glob.glob(files_hist_y)
                
                
            else:
                
                dir_lens = '/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/month_1/'
                files_hist_x = dir_lens+cvar_x+'/'+cname+'.cam.h0.'+cvar_x+'.18*.nc'
                files_hist_y = dir_lens+cvar_y+'/'+cname+'.cam.h0.'+cvar_y+'.18*.nc'
                files_ls_x  = glob.glob(files_hist_x)
                files_ls_y  = glob.glob(files_hist_y)
                
         
            
            print('- Grabbing file(s) for LENS1/2 (CESM1/2)')
           
            da_sst = xr.open_mfdataset(files_ls_x,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")[cvar_x]
            
            if ctype=='cesm1' and cvar_comp == 'PRECT': # Need to grab PRECC and PRECL files separately 

                files_hist_pc = files_hist_y.replace('PRECT', 'PRECC')
                files_hist_pl = files_hist_y.replace('PRECT', 'PRECL')

                files_ls_pc  = glob.glob(files_hist_pc)
                files_ls_pl  = glob.glob(files_hist_pl)
                
                da_pc = xr.open_mfdataset(files_ls_pc,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")['PRECC']
                da_pl = xr.open_mfdataset(files_ls_pl,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")['PRECL']
                
                da_y = da_pc+da_pl
                
            else:
                da_y = xr.open_mfdataset(files_ls_y,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")[cvar_y]
              
                
        
            vscale = cvar_scale
    
        case _ if ctype in 'cesm3':
    
    
            # add vscale here
            vscale = cvar_scale
            
            if os.path.exists(ofile_x) and os.path.exists(ofile_y) and lread_ts_file:
    
                print('- Existing files exist for the case - using them')
            
                print('     ',ofile_x)
                print('     ',ofile_y)
            
                da_x = xr.open_dataset(ofile_x)[cvar_x]
                da_y = xr.open_dataset(ofile_y)[cvar_y]
                
                # Pick the right directory (hannay/gmarques)
    
                dir_c3 = '/glade/derecho/scratch/hannay/archive/'
                
                if os.path.isdir(dir_c3+cname):
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
                        file_ls = dir_c3+cname+'/atm/hist/'+cname+'.cam.h0a.'+yr_str+'*.nc'
                        files_hist.extend(glob.glob(file_ls)) 
    
                    files_hist.sort()
                
            else:
                file_hist = dir_c3+cname+'/atm/hist/'+cname+'.cam.h0a.*.nc'
                    
               
     
     # File checks           
                
                print('-Reading ',len(files_hist),' files  (first/last)')
                print('   -',files_hist[0])
                print('   -',files_hist[-1])
                
            # Open them as multiple files
    
                print('- Slow read of h0a output...')
                ds_cesm = xr.open_mfdataset(files_hist,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")
                
                da_x = ds_cesm[cvar_x]
                da_y = ds_cesm[cvar_y]
    
                print('-Done')
                
            # Write out the timeseries of 2D field files?
            
                if lwrite_ts_file: 
    
                    print("- Write out files of 2D fields SST/VAR")
        
                    print('    ',ofile_x)
                    print('    ',ofile_y)
        
        
                    da_x.to_netcdf(ofile_x,mode="w")
                    da_y.to_netcdf(ofile_y,mode="w")
        
                    da_x.close()
                    da_y.close()
    
                    print('-Done')

        case _:
            
            print("No ctype match for "+case)

    return darray_x,darray_y





    