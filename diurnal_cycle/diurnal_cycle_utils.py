import xarray as xr
import numpy as np

import matplotlib.pyplot as plt


import glob as glob
import os as os
import re


    
    

'''
	BLOCKING UTILIY ROUTINES
'''




'''
    dc_data - Reads in and processes data beofre sending back to be operated on
    dc_freq1d - Calculates 1D blocking frequencies (longitude) - D'Andrea et al. ƒ(1998)

'''

import xarray as xr
import pandas as pd
import numpy as np

import importlib
import sys
import pprint
import time


# Output directory for output files (and to read them in).
fout_dir = '/glade/u/home/rneale/python/python-netcdf/diurnal_cycle/'






############################################
#           CASE(S) SETUP                  #
############################################

def case_setup (case_names,case_types,ystart,yend):

    all_case_info = {}
    
    cesm_pref_names = ['f40','f.','b.']
    dir_cesm_all = ['/glade/derecho/scratch/rneale/archive/']

  
    
    for icase,case_name in enumerate(case_names):

        case_type = case_types[icase]
        
        match (case_type):
               
        
            case ('obs'):
        
                match (case_name):
                    case ('TRMM'):
                        dir_case0 = '/glade/campaign/cgd/amp/rneale/data/TRMM/3hrly/1deg/'
            case ('SAAG'):
                
                match (case_name):
                    case ('f.e22r.SAMwrf01.ne30x1.L32.REFERENCE'):
                       dir_case0 = ''
    
                
    
    # Conmstruct meta datafarme
    
        all_case_info[case_name] = [case_type,ystart[icase],yend[icase],case_names[icase],dir_case0]
    
    
#    pprint.pprint(all_case_info)
    
    case_info = pd.DataFrame.from_dict(all_case_info, orient='index',columns=['Case type','Start Year','End Year','Run Name','Dir Loc.'])
    #    df_info = pd.DataFrame(data=all_ens_info)
    display(case_info)
    
    
    return case_info



















############################################
# Set ensemble/single/obs case information #
############################################


def find_data_info(data_desc,data_name,ystart,yend):
    

    import lens_simulations as sim_names
    importlib.reload(sim_names)

    obs_sources = ['ERA5','TRMM','','GPCP','CPC']

    fname = '-> find_data_info -> '
    
    all_ens_info = {}    

    
# Loop ensemble sets (ensembles/obs/singlecases)
    
    for iens,ens_name in enumerate(ens_names):
        
        
        if ens_name in ['CESM1','CESM2','E3SMv1','E3SMv2','EAMv2','CAM6']:
            run_names = sim_names.get_ens_set_names(ens_name,mem_num[iens])
        else:
            run_names = [ens_name]
    
        
        match (ens_name):
            case 'CESM1':
                
                ens_type = 'model'
                dir_ens0 = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/' 
                dir_day_add = 'daily'

                file_templates = [(dir_ens0+dir_day_add+'/VAR_TBD/'+this_run+'.cam.h1.VAR_TBD.19200101-20051231.nc') for this_run in run_names]      
                
                # Need to modify start date for CESM1 ens# 1. 
                run_ens1 = 'b.e11.B20TRC5CNBDRD.f09_g16.001'
                if run_ens1 in run_names:
                    file_templates[run_names == run_ens1]=file_templates[run_names == run_ens1].replace('1920','1850',1)
                                
            case 'CESM2':
                
                # CESM2 is tricky for the files.
                # Place hold for now and grab the date specific files later.
                ens_type = 'model'
                dir_ens0 = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/'
                dir_day_add = 'day_1'
                file_templates = [(dir_ens0+dir_day_add+'/VAR_TBD/'+this_run+'.cam.h1.VAR_TBD.DATE_RANGE.nc') for this_run in run_names]    

            case 'E3SMv1':
                
                ens_type = 'model'
                dir_ens0 = '/glade/campaign/cgd/amp/rneale/e3sm/'
                dir_day_add = 'day_1'
                file_templates = [(dir_ens0+this_run+'/tseries/'+this_run+'_dmeans_ts_VAR_TBD.nc') for this_run in run_names]
                
            case ens_name if ens_name in ['E3SMv2','EAMv2','CAM6']:
                
                ens_type = 'model'

                if ens_name == 'CAM6':   # CAM6 different locations
                    cmodel = 'cam'
                    dir_ens0 = '/glade/campaign/cesm/development/cvcwg/cvwg/f.e21.FHIST_FSSP370_BGC.f09_f09.ersstv5.goga/'
                else:
                    cmodel = 'eam'
                    dir_ens0 = '/glade/campaign/cgd/ccr/E3SMv2/FV_regridded/'

                
                dir_day_add = 'day_1'


                
                file_templates = [(dir_ens0+this_run+'/atm/proc/tseries/'+dir_day_add+'/'+this_run+'.'+cmodel+'.h1.VAR_TBD.DATE_RANGE.nc') for this_run in run_names]
                
               
                # Modify filename at range accordingly
                if ens_name=='EAMv2' : file_templates=[fl.replace('DATE_RANGE','19760101-20141231',1) for fl in file_templates]
                if ens_name=='E3SMv2': file_templates=[fl.replace('DATE_RANGE','18500101-20141231',1) for fl in file_templates]
                if ens_name=='CAM6'  : file_templates=[fl.replace('DATE_RANGE','18800101-20150101',1) for fl in file_templates]

            case ens_name if 'b.e30' in ens_name: # Individual CESM3 development run cases

                cmodel = 'cam'
                ens_type = 'model'
                dir_ens0 = '/glade/derecho/scratch/rneale/archive/'
#                file_templates = [(dir_ens0+this_run+'/atm/hist/'+this_run+'.'+cmodel+'.h2a.*.nc') for this_run in run_names]
                file_templates = [(dir_ens0+this_run+'/tseries/'+this_run+'_dmeans_ts_Z500.nc') for this_run in run_names]    
                ens_name = 'CESM3'
        
            case ens_name if ens_name in obs_sources:
                
                ens_type='obs'
                dir_ens0 = '/glade/work/rneale/data/'+ens_name+'/'
                file_templates = [dir_ens0+'VAR_TBD.day.mean.nc']
                
            case _  : 
                
                print(' ')
                print(ens_name+' is not a recognized case or ensemble set')
                sys.exit(0)       

# Loop over ensembles to get the file to be read in.

            
        all_ens_info[ens_name] = [ens_type,mem_num[iens],ystart[iens],yend[iens],run_names,file_templates]  

    
#    pprint.pprint(all_ens_info)

    df_info = pd.DataFrame.from_dict(all_ens_info, orient='index',columns=['Ensemble Type','Ensemble Size','Start Year','End Year','Run Name','Run File'])
#    df_info = pd.DataFrame(data=all_ens_info)
    display(df_info)
    
    return df_info



















###################################
# Read in data for analysis
###################################

# A little tricky as we don't want to read in the whole dataset first

def caseinfo_get(dc_meta,var_name,season,diag_set):

    fname = '-> dataset_get -> '

    tstart = time.time()
    
    case_names = list(dc_meta.index)

    # Request info.

    print(fname+ 'Requested season : ',season)


    ldcexist_try = False
    # Final dataset dictionary
    ds_cases = {}

    
    # Loop ensemble sets to setup datasets

    for icase,case_name in enumerate(case_names):
                        
        print(case_name)     
        
        year_start = dc_meta.loc[case_name]['Start Year']
        year_end = dc_meta.loc[case_name]['End Year']      

        print(fname,'Case name',case_name)
        print(fname+ 'Requested year range : ',year_start,'->',year_end)

        # Chunk sizes
        chunk_sizes = {'time': 365, 'latitude': 360, 'longitude': 180}

        # Check to see if case data exists post processed as lat/lon/dcycle_hrs?
        
        if ldcexist_try:
            print(fname + 'Checking to see if processed data already exists')
            
            match (case_name):
                case 'SAAG':
                    dir_data =  '/glade/campaign/cgd/amp/patc/SAM_PostProc/Data_CESM/MEANS/'
                    case_dir  = '/L32/ne30x1/h2/'
                    files_all = dir_data+case_dir
                    print(fname + 'Directory -> ' + dir_data)
                    
                    
             



        
        else :
        
            match (case_name):   
                case 'TRMM':
                    dir_data = '/glade/campaign/cgd/amp/rneale/data/TRMM/3hrly/1deg/'
                    files_all = dir_data+'3B42.??????.3hr_V7.1x1.nc'
                    flist = sorted(glob.glob(files_all))
                    print(fname + 'Directory -> ' + dir_data)
                    print(fname + 'Total # of files -> ' , len(flist))
                    print(fname + 'First file -> ' + os.path.basename(flist[0]))
                    print(fname + 'Last file -> ' + os.path.basename(flist[-1]))
                    
     ## Subset for years and months
    
                    flist = [ff for ff in flist if '2010' in ff]
                    print(fname + 'Total # of files -> ' , len(flist))
                    print(fname + 'First file -> ' + os.path.basename(flist[0]))
                    print(fname + 'Last file -> ' + os.path.basename(flist[-1]))
    
                    flist = [ff for ff in flist if re.search(r'\d{4}01', ff)]
                    print(fname + 'Total # of files -> ' , len(flist))
                    print(fname + 'First file -> ' + os.path.basename(flist[0]))
                    print(fname + 'Last file -> ' + os.path.basename(flist[-1]))
    
            
    
                    


    
    print(fname,f'Function Duration: {time.time() - tstart}') ; print()
    
    return flist
    



###################################################


def calc_dcycle(case_ds):

# Loop over cases and produce a 2Dxtime of day array that can be written to a file and or plotted
# Each case also read in a diurnal cycle dataset from a previous writtten out calculation.
    
    
    case_ds
    
    nfiles = len()
    
    # Open CESM file (replace with actual file name)
    ds = xr.open_dataset("your_file.nc")
    
    # Choose variable (change 'PRECT' to your target variable)
    var_name = "PRECT"
    data = ds[var_name]
    
    # Convert time coordinate to hour-of-day
    hour_of_day = data['time'].dt.hour
    
    # Compute mean diurnal cycle
    diurnal_cycle = data.groupby(hour_of_day).mean(dim="time")
    
    # Apply FFT to get harmonics
    fft_result = np.fft.fft(diurnal_cycle)
    
    # Compute frequencies (0-23 hours, since we have 24 hours in a diurnal cycle)
    freqs = np.fft.fftfreq(len(diurnal_cycle), d=1)  # d=1 since it's hourly data
    
    # Compute amplitude and phase of harmonics
    amplitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # Plot Harmonic Amplitudes
    plt.figure(figsize=(8, 5))
    plt.stem(freqs[:12], amplitude[:12], basefmt=" ")
    plt.xlabel("Harmonic Number")
    plt.ylabel("Amplitude")
    plt.title(f"Harmonics of {var_name} Diurnal Cycle")
    plt.grid()
    
    
























#################################################################################
#    Logic for reading/writing files with block related values for this ensemble 
#################################################################################
    
def block_file_read_write(ens_name,nens,year_start,year_end,bseason,block_array_ens,block_diag,file_opts):


    fname = '-> block_file_read_write -> '   
   
            
    file_netcdf = 'block_' + block_diag +'_' + ens_name + '_' + 'nens.'+ str(nens) +'_'+ year_start+ '-' + year_end + '_' +bseason+'.nc' 
    file_data = fout_dir+file_netcdf
  
    
    match(file_opts):
    
        case 'w' : 
           
            print(fname,'Writing file for ensemble ' ,ens_name,' = ',file_netcdf)  
            block_array_ens = block_array_ens.rename('BLOCK_'+block_diag)
            block_array_fout = block_array_ens.to_dataset()

            block_array_fout.to_netcdf(file_data)
            print(fname,'Done ...')
            
            return block_array_ens # Just for pass through back to main routine
            
        case 'r' : 
            
            print(fname,'Reading file for ensemble ' ,ens_name,' = ',file_netcdf)
            block_ens_fin =  xr.open_dataset(file_data)['BLOCK_'+block_diag]
          

            print(fname,'Done ...')
            
            return block_ens_fin

        case 'x' :
    
            print(fname,'No date read/write for ' ,ens_name)

            return block_array_ens # Just for pass through back to main routine

        case _ :
            
            print(fname,'Unknown read/write options - should be r,w or x ' ,ens_name)
            sys.exit(0)       

    




















































