
############################################
# SCAM name/LES name/LES units/scale -> LES
############################################

# Tried to match the variable names as much as possible from here
# /glade/p/mmm/nmmm0058/patton/anl/3d

import metpy.constants as mconst
import pandas as pd


### Constants ###
cp_air = mconst.dry_air_spec_heat_press.magnitude # Specific heat for dry air
Lv = mconst.water_heat_vaporization.magnitude 

var_r = {}  # Long name/scam variable/units/scale

### COORDS ###
var_r['ts']         =   ['Simulation start time','','[s]',1.]
var_r['cs']         =   ['Chemistry start time','','[s]',1.]          
var_r['latitude']   =   ['Latitude','lat','[deg]',1.]           
var_r['longitude']   =  ['Longitude','lon','[deg]',1.]   
var_r['zu']   = ['Height of grid at grid center','','[m]',1.]


var_r['zw']   = ['Height of grid at grid upper-face','','[m]',1.]
var_r['p']   = ['Pressure','','[Pa]',1.]

### CONSTANTS ###
#var_r['grav'] = ['Gravitational acceleration','','[m/s^2]',1.]
#var_r['p_ref'] = ['Reference pressure','','[pa]',1.]

#var_r['t_ref'] = ['Reference temeperature','','[pa]',1.]
#var_r['ustar'] = 
#var_r['wstar']   =       
#var_r['omonin']   = 



### BASIC VARIABLES ###
var_r['wqsfc']   = ['Surface water vapor specific humidity flux','LHFLX','[m/s kg/kg]',1./Lv]  
var_r['wtsfc']   = ['Surface potential temperature flux','SHFLX','[mK/s]',1./cp_air] 
var_r['t']   = ['Potential Temperature','','[K]',1.] 
var_r['tv'] = ['Virtual Potential Temperature','','[K]',1.] 
var_r['q']   = ['Specific Humidity','Q','[mK/s]',1.] 
var_r['zi_t']   = ['Height of maximum potential temperature gradient','','[m]',1.] 
var_r['zi_q']   = ['Height of maximum specific humidity gradient','','[m]',1.]



### VARIANCES/CO-VARIANCES ###
##var_r['tke_r']  = ['Resolved TKE','','[m^2/s^2]',1.]
var_r['ww_r']   = ['Resolved ww-covariance','WP2_CLUBB','[m^2/s^2]',1.]
var_r['uu_r']   = ['Resolved uu-covariance','UP2_CLUBB','[m^2/s^2]',1.]
var_r['vv_r']   = ['Resolved vv-covariance','VP2_CLUBB','[m^2/s^2]',1.]
var_r['tt_r']   = ['Resolved pot. temp. variance','THLP2_CLUBB','[K^2]',1.] 


var_r['wt_r']   = ['Resolved vertical pot. temp. flux','WPTHLP_CLUBB','[mK/s]',1./cp_air]
var_r['qq_r']   = ['Resolved specific humidity variance','RTP2_CLUBB','[(kg/kg)^2]',1.]
var_r['wq_r']   = ['Resolved vertical spec. hum. flux','WPRTP_CLUBB','[m/s kg/kg]',1./Lv]

var_r['www_r']  = ['Resolved www triple-moment','WP3_CLUBB','[m^3/s^3]',1.]


#var_r['missing_data'] = ['Missing data','','[-]',-999.]            
# Constajnts = [grav,p_ref,t_rtef,rd,rv,eps,vk,mwair,mwh20,missing_data]

# Time should come last (as it is resetting the time array coord. to that required by LES/SAS)
var_r['time']   =       ['Time','','[s]',1.]  

### REMAINING/NOT POSSIBLE

#var_r['u']   = 
#var_r['v']   = 
#var_r['w']   = 

#var_r['uu_r']   = 
#var_r['uv_r']   = 
#var_r['uw_r']   = 
#var_r['vv_r']   = 

#var_r['uuu_r']   = 
#var_r['vvv_r']   = 

#var_r['ttt_t']   = 
#var_r['qqq_r']  =  ['Resolved www triple-moment','RTP3_CLUBB','[(kg/kg)^3]',1.]

#var_r['wtv_r']   = ['Resolved www triple-moment','WP3_CLUBB','[m^2/s^2]',1.]
#var_r['tttv_r']   = ['Resolved www triple-moment','WP3_CLUBB','[m^2/s^2]',1.]
#var_r['omonin']   = 






#---------- Pandas display frame -----------#


pd.set_option('display.width', 1000)
var_r_df = pd.DataFrame.from_dict(var_r, orient='index',
                                       columns=['Long Name','SCAM Name','Units','Scaling'])
var_r_df.style.set_properties(**{'background-color': 'black',                                                   
                                    'color': 'lawngreen',                       
                                    'border-color': 'white'})
#--------- END FUNCTIONS ---------#