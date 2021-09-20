#########################################

# SCAM COMPUTE FUNCTIONS

#########################################

print('+++ IMPORTING UTILS FUNCTIONS +++')

### Imports ###
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import scipy as spy
import pandas as pd
import datetime as dt
import cftime as cft
from nc_time_axis import CalendarDateTime
import subprocess as sp	

import metpy.constants as mconst
import metpy.calc as mpc
	


#### CONSTANTS USED BELOW ####


r_gas = mconst.dry_air_gas_constant.magnitude   # Specific gas constant for dry air (kg/joule)
cp_air = mconst.dry_air_spec_heat_press.magnitude # Specific heat for dry air
Lv = mconst.water_heat_vaporization.magnitude       # Latent heat of vaporization
grav = mconst.earth_gravity.magnitude

p0 = 100.*mconst.pot_temp_ref_press.magnitude # P0 but in Pa

r_cp = r_gas/cp_air    # r/cp


    
######################################################
### PBL Height Calculations Based on q/th gradient ###
######################################################
	
	
def pbl_grad_calc(var_grad,scam_in):
	
# Var grad -> variable to be used gradient (Q or TH(eta))
    pres,zhgt = vcoord_scam('mid',scam_in)
    var_in = scam_in.T*(p0/pres)**r_cp if var_grad in 'TH' else scam_in.Q

# Gradient wrt height not pressure
    var_in['lev'] = zhgt[0,:].values # Add height instead of pressure for this variable (BE CAREFUL)           
    dvardz = var_in.differentiate("lev") # Find field gradient wrt HEIGHT (over limited region)	
            

# Locate gradient maximum ain a bounded regin            
    ztop_mask = 3000 ; zbot_mask = 100  # Restrict to approx region of PBL top
    dvardz = dvardz.where((dvardz.lev < ztop_mask) & (dvardz.lev > zbot_mask)) # MASK IN LOCAL PBL REGION
    dvardz_kmin = dvardz.argmax(axis=1) if var_grad in 'TH' else dvardz.argmin(axis=1) # Find the index of the max/min in the vertical

# Map to a height level    
    
    var = zhgt.isel(lev=dvardz_kmin)

	# For compatability with more scripts   
    if 'lat' in list(var.dims): var = var.isel(lat=0,lon=0)

    return (var)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
######################################################
###  Vertical Coordinates From SCAM                ###
######################################################	
	    
# Grabs vertical profile of pressure interfaces and mid-levels and converts them to height for plotting
    
    
def vcoord_scam(imlev,scam_in):

 
    
    plevm = scam_in['hyam']*p0 + scam_in['hybm']*scam_in['PS'] # Mid level
    plevi = scam_in['hyai']*p0 + scam_in['hybi']*scam_in['PS'] # Interface level
    
 # For compatability with more scripts   
    if 'lat' in list(plevm.dims):
        plevm = plevm.isel(lat=0,lon=0)
        plevi = plevi.isel(lat=0,lon=0)
    
    plevm = plevm.transpose('lev','time')  # Force this ordering for constructing profiles (some cases differ)
    plevi = plevi.transpose('ilev','time')
    
    plevm.attrs['units'] = "Pa"
    plevi.attrs['units'] = "Pa"

# Height with standard atmosphere

#    print(scam_in)
    zlevm = plevm
#    print(1000.*mpc.pressure_to_height_std(plevm).values)
    zlevm_vals = 1000.*mpc.pressure_to_height_std(plevm).values
    zlevi_vals = 1000.*mpc.pressure_to_height_std(plevi).values
    dzbot = np.array(1000.*mpc.pressure_to_height_std(plevi[-1,:]).values)
   
    
#    print(plevi[-1,:])

    zlevm = plevm.copy(deep=True) # use plev variable structures on the dataset
    zlevi = plevi.copy(deep=True)
    

    zlevm.values = zlevm_vals # Just move across values to retain xarray wrapper
    zlevi.values = zlevi_vals # ""
    
    nlevm = scam_in['lev'].size # Size of vertical dimensions
    nlevi = scam_in['ilev'].size
        
# Normalize to ilev bottom being Z of surface from an expanded 2D array, then transpose back to original set up.
    
    dzbot_2d = np.tile(dzbot,(nlevm,1))  
    zlevm = np.subtract(zlevm,dzbot_2d)
    zlevm = zlevm.transpose()
    
    dzbot_2d = np.tile(dzbot,(nlevi,1))  
    zlevi = np.subtract(zlevi,dzbot_2d)
    zlevi = zlevi.transpose()
    
    
    
        
    v_coord = [plevm,zlevm] if imlev in 'mid' else [plevi,zlevi] # Return dep. on interface/mid
        
    return v_coord





######################################################
###  Derived Variable for Plotting in SCAM.        ###
######################################################	
	
def dev_vars_scam(var_name,sfile_in):
    
    plevm = sfile_in['hyam']*p0 + sfile_in['hybm']*sfile_in['PS'].isel(lat=0,lon=0)
    if 'lat' in list(plevm.dims): plevm = plevm.loc(lat=0,lon=0) # For compatability with more scripts
   
    if var_name in ['TH','THL','THV'] : 
        theta = sfile_in['T'].isel(lat=0,lon=0).transpose()*(p0/plevm)**r_cp #
        theta.attrs['long_name'] = "Potential Temperature" 
        theta.attrs['units'] = "K" 
        dev_var = theta
  
   
    if var_name=='THV' : # (For dry air)
        thetav = theta*(1.+0.61*sfile_in['Q'].isel(lat=0,lon=0).transpose())
        thetav.attrs['long_name'] = "Virtual Potential Temperature" 
        thetav.attrs['units'] = "K" 
        dev_var = thetav   
        
    if var_name =='THL': 
        thetal = dev_var-(theta/sfile_in['T'].isel(lat=0,lon=0).transpose()) \
            *(Lv/cp_air)*sfile_in['Q'].isel(lat=0,lon=0).transpose() 
        thetal.attrs['long_name'] = "Liq. Water Potential Temperature"
        dev_var = thetal
    return (dev_var)




####################################################################
###  Interpolate NCAR LES on rgular grid with 1D coordinates    ###
####################################################################	
	
def les_reg_grid(var_les,plev,zlev):
    # Set coordinate levels of z as the mean of the z array
    print(zlev[0:20,0:150])
    print(var_les)
    
#    pvarp_sm = pvarp_sm.values.ravel()
#    plev = plev.values.ravel()
#    zlev = zlev.values.ravel()
#    hour_frac = np.repeat(hour_frac,np_les)
   
    return (var_les_reg)








#################################
### Local Time Offset from GMT
#################################


""" GET THE ZOFFSET """
""" AND FIGURE OUT THE DAYS TO GRAB """

def get_iop_info(case_iop):
  

    if case_iop == 'SAS': 
        zoffset = -6
        iop_day = -999.
        ttmin = 5. ; ttmax = 18
    if case_iop == 'PERDIGAO': 
        zoffset = 2. # CET?
        iop_day = '2017-05-23'
        ttmin = 5. ; ttmax = 20
    if case_iop == 'RICO' : 
        zoffset = -4. ##  ??
#        iop_day = '2004-12-16'
        iop_day = -999.
        ttmin = 5. ; ttmax = 20

    
    return(zoffset,iop_day,ttmin,ttmax)


#### 







####################################################################
###  OPEN AND READ SCAM OR LES IOP FILES.                        ###
####################################################################	


def scam_open_files(case_iop,file_num,run_name,dir_main):

    
# SCAM Output
 
    if file_num != 'LES' :
        
        
# File assignment using core dir from notebook
        
        files_dir = dir_main+'history/'
        
        files_pre = files_dir+'FSCAM.T42_T42.'+case_iop+'.' 
        
# NEED TO CLEAN UP THIS FILE LOGIC TO BE MORE SPECIFIC
        files_list = files_pre+file_num+'.cam.h0*00.nc'
  
        print(files_list)

# Grab files (could be more than one)
#        print(sp.run(['ls',files_list], shell=True)) # Fix tis check to work

        try:
            stdout1 = sp.check_output('ls '+files_list, shell=True)
        except sp.CalledProcessError as e:
            print(case_iop+' - '+run_name+' - Files Not Present - EXITING...')

# Call to OS ls command

        files_in = sp.getoutput('ls '+files_list)
                
# Convert string to an array    
    
        files_in = files_in.split()
                   
# Grab some iop specific info.
            
        zoffset,iop_day,ttmin,ttmax = get_iop_info(case_iop)
    
# Check for and concatonate multiple files
# Multi
        if len(files_in) > 1:
           
            iop_dset = xr.open_mfdataset(files_in)
                        
        else :
            
# Single
            iop_dset = xr.open_dataset(files_in[0],engine='netcdf4',use_cftime=True) 
#            iop_dset = xr.decode_cf(iop_dset)

#            time = cft.date2num(iop_dset.time, "hours", calendar='noleap')
#            print(time)
# Trim to desired data
        if iop_day != -999.: iop_dset = iop_dset.sel(time=iop_day)  # Choose day from specs. (May, 23 2017)
        
    
        iop_dset = iop_dset.load() # Load this into memory as dask does not allow indexing or something like that.  
        
  
    if file_num == 'LES' :
            
# LES Output
        les_file_dir = dir_main+'LES/'
        les_file_in = les_file_dir+run_name+'_'+case_iop+'_LES.nc' 

        
        iop_dset = xr.open_dataset(les_file_in,engine='netcdf4') 
         
       
    
    return(iop_dset)





####################################################################
### GET LES DATA FROM FILES (currently on NCAR nc file from Ned) ###
####################################################################	


def get_les_dset(les_dset,plot_df,var):

####### LES Specific #######
         

    lscale = plot_df.loc[var,'lscale']
    les_tstart = les_dset['ts'] # Start time (local?) seconds after 00
    les_time = les_dset['time'] # Time (from start?) in seconds

    les_toffset = 0. # Strange time stuff in LES?
    hour_frac_les = (les_tstart+les_time)/3600.+les_toffset  # Transform into hours
        

## Variable ##

    var_les = plot_df.loc[var,'var_les']      
    data_les = les_dset[var_les] # Read in data # 
    data_les = data_les*lscale

    plev_les = les_dset['p']
    plev = 0.01*plev_les
    zlev = les_dset['zu']

    les_pvarp = data_les

    
## Set coordinate for data
    
    nz_les = les_pvarp.sizes['nz']
    zlev = zlev[1,0:nz_les]

    
## RETURN INFO FOR. PLOTTING
    return(les_pvarp,hour_frac_les,zlev)

