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