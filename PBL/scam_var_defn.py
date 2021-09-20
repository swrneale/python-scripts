####################################################################
###  VARIABLE SETUPS FOR PLOTTING (cotour,ranges, etc)           ###
####################################################################	

print('+++ IMPORTING VAR DEFINITION FUNCTIONS +++')

import pandas as pd
import metpy.constants as mconst
import metpy.calc as mpc

### Constants ###

r_gas = mconst.dry_air_gas_constant.magnitude   # Specific gas constant for dry air (kg/joule)
cp_air = mconst.dry_air_spec_heat_press.magnitude # Specific heat for dry air
Lv = mconst.water_heat_vaporization.magnitude       # Latent heat of vaporization
grav = mconst.earth_gravity.magnitude

p0 = 100.*mconst.pot_temp_ref_press.magnitude # P0 but in Pa

r_cp = r_gas/cp_air    # r/cp




def var_plot_setup (plot_set,all_vars_info) :

    plot_dic = {}


# 1D TIMESERIES
    if (plot_set=='1d_ts') :
    # VAR -> vscale/ymin/ymax/p_var/lscale
        plot_dic['LHFLX']  = ['Latent Heat Flux','W/m2',1.,0.,400,'wqsfc',Lv]
        plot_dic['SHFLX']  = ['Sensible Heat Flux','W/m2',1.,0., 300,'wtsfc',cp_air]
        plot_dic['TS']     = ['Surface Temperature','K',1., 290., 310.,'',1.]
        plot_dic['PBLH']   = ['Boundary Layer Depth (CAM)','meters',1., 0., 2800.,'zi_t',1.] # zi_t: height of max theta gradient
        plot_dic['PBLH_DTH'] = ['Boundary Layer Depth (dth/dz max)','meters',1., 0., 2500.,'zi_t',1.] # zi_t: height of max theta gradient
        plot_dic['PBLH_DQ'] = ['Boundary Layer Depth (dq/dz max)','meters',1., 0., 2500.,'zi_q',1.] # zi_t: height of max q gradient
        plot_dic['PBL_DQMAX'] = ['Boundary Layer dq/dz Max.','g/kg/km',1000.*1000, -100, 0.,'q',1.] # Min value of dq/dz
        plot_dic['PRECL']  = ['Large-Scale Precipitation','mm/day',86400.*1000., 0., 10.,'',1.]
        plot_dic['PRECC']  = ['Convective Precipitation','mm/day',86400.*1000., 0., 10.,'',1.]
        plot_dic['FLNS']   = ['Surface Net Short-wave Radiation','W/m2',1., 200., 800.,'',1.]
        plot_dic['CAPE']   = ['CAPE','J/kg',1., 0., 800.,'',1.]

        
        plot_df = pd.DataFrame.from_dict(plot_dic, orient='index',
            columns=['long_name','units','vscale','ymin','ymax','var_les','lscale'])
        
        
        
        
        
        
        
# 2D TIMESERIES 
        
    if (plot_set=='2d_ts') :
        plot_dic['T'] = ['Temperature', 
        1.,260.,305.,-.5,.5,'',1.,'K']

        plot_dic['RELHUM'] = ['Relative Humidity',
            1.,10., 120.,-10.,10.,'',1.,'%']
        plot_dic['CLOUD'] = ['Cloud Fraction', 
            100., 0., 100.,-10.,10.,'',1.,'%']
        plot_dic['Q'] = ['Specific Humidity',
            1000., 1., 12.,-2,2.,'q',1000.,'g/kg']

        plot_dic['TH'] = ['Potential Temperature', \
            1., 295, 305.,-.5,.5,'t',1.,'K']
        plot_dic['THL'] = ['Liquid Water Potential Temperature', \
            1., 270, 310.,-1.,1.,'thl',1.,'K']
        plot_dic['THV'] = ['Virtual Potential Temperature', \
            1., 295, 305.,-2.,2.,'tv',1.,'K']

        plot_dic['DCQ'] = ['Humidity Tendencies - Moist Processes', \
            1000., -5., 5.,-1.,1.,'',1.,'g/kg/day']
        plot_dic['DTCOND']  = ['Temperature Tendencies - Moist Processes', \
            86400., -10., 10.,-1.,1.,'',1.,'K/day']   

        plot_dic['ZMDT'] = ['Temperature Tendencies - Deep Convection', \
            86400., -10., 10.,-1.,1.,'',1.,'K/day']
        plot_dic['ZMDQ'] = ['Humidity Tendencies - Deep Convection', \
            86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']

        plot_dic['CMFDT'] = ['Temperature Tendencies - Shallow Convection', \
            86400., -10., 10.,-10.,10.,'',1.,'K/day']
        plot_dic['CMFDQ']  = ['Humidity Tendencies - Shallow Convection', \
            86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']



    # BL/Turb. vars.                             
        plot_dic['DTV']  = ['Temperature Tendencies - Vertical Diffusion',
            86400., -15., 15.,-1.,1.,'',1.,'K/day']
        plot_dic['VD01']  = ['Humidity Tendencies - Vertical Diffusion',
            6400.*1000., -50, 50.,-1.,1.,'',1.,'g/kg/day']

        plot_dic['STEND_CLUBB']    = ['Temperature Tendencies - CLUBB', \
            86400./1000., -20, 20,-6.,6.,'',1.,'K/day'] #J/kg.s -> K/day
        plot_dic['RVMTEND_CLUBB']  = ['Humidity Tendencies - CLUBB', \
            1000.*86400, -100., 100.,-20.,20.,'',1.,'g/kg/day']

        plot_dic['WPRTP_CLUBB'] = ['w,q - Flux Covariance - CLUBB', \
            1., -0., 600.,-100.,100.,'wq_r',Lv,'W/m^2']
        plot_dic['WPTHLP_CLUBB'] = ['w,thl - Flux Covariance - CLUBB', \
            1., -100., 100.,-10.,10.,'wt_r',cp_air,'W/m^2']
        plot_dic['WPTHVP_CLUBB'] = ['w,thv - Flux Covariance - CLUBB', \
            1., -100., 100.,-5.,5.,'',1.,'W/m^2']

        plot_dic['THLP2_CLUBB'] = ['T^2 - Variance - CLUBB', \
            1., 0., 0.05,-0.01,0.01,'tt_r',1.,'K^2']   
        plot_dic['RTP2_CLUBB']  = ['q^2 - Variance - CLUBB', 1., 0., 2.5,-0.5,0.5,'qq_r',1000.*1000.,'g^2/kg^2']
    
    
    
        plot_dic['WP2_CLUBB']   = ['w^2 - Variance - CLUBB', \
            1., 0., 2.,-0.5,0.5,'ww_r',1.,'m^2/s^2'] 

        plot_dic['WP3_CLUBB']   = ['w^3 - Skewness  - CLUBB', \
            1., 0., 0.5,-0.05,0.05,'www_r',1.,'m^3/s^3']
        
#### TURN INTO DATFRAME ####
        
        plot_df = pd.DataFrame.from_dict(plot_dic, orient='index', \
            columns=['long_name','vscale','cmin','cmax','acmin','acmax','var_les','lscale','units'])

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# 1D SNAPSHOT PROFILE PLOTS

    if (plot_set=='1d_snap') :
        
        plot_dic['T']      = ['Temperature',1.,260.,305.,-20.,20.,'',1.,'K']
        plot_dic['RELHUM'] = ['Relative Humidity',1.,10., 120.,-100.,100.,'',1.,'%']
        plot_dic['CLOUD']  = ['Cloud Fraction',100., 0., 100.,-80.,80.,'',1.,'%']
        plot_dic['Q']      = ['Specific Humidity',1000., 1., 12.,-5,5.,'q',1000.,'g/kg']
 
        plot_dic['TH']  = ['Potential Temperature',1., 295, 305.,-20.,20.,'t',1.,'K']
        plot_dic['THL']  = ['Liquid Water Potential Temperature',1., 270, 310.,-20.,20.,'thl',1.,'K']
        plot_dic['THLV']  = ['Virtual Potential Temperature', 1., 270, 310.,-20.,20.,'tv',1.,'K']
    
        plot_dic['DCQ']  = ['Humidity Tendencies - Moist Processes',1000., -5., 5.,-5.,5.,'',1.,'g/kg/day']
        plot_dic['DTCOND']  = ['Temperature Tendencies - Moist Processes',86400., -10., 10.,-10.,10.,'',1.,'K/day']   
    
        plot_dic['ZMDT']  = ['Temperature Tendencies - Deep Convection',86400., -10., 10.,-10.,10.,'',1.,'K/day']
        plot_dic['ZMDQ']  = ['Humidity Tendencies - Deep Convection',86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']
    
        plot_dic['CMFDT']  = ['Temperature Tendencies - Shallow Convection',86400., -10., 10.,-10.,10.,'',1.,'K/day']
        plot_dic['CMFDQ']  = ['Humidity Tendencies - Shallow Convection',86400.*1000., -2, 2.,-8.,8.,'',1.,'g/kg/day']
   
    
                              
# BL/Turb. vars.                             
        plot_dic['DTV']  = ['Temperature Tendencies - Vertical Diffusion',86400., -30., 13.,-10.,10.,'',1.,'K/day']
        plot_dic['VD01']  = ['Humidity Tendencies - Vertical Diffusion',86400.*1000., -50, 50.,-8.,8.,'',1.,'g/kg/day']
    
        plot_dic['STEND_CLUBB']    = ['Temperature Tendencies - CLUBB',86400./1000., -20, 20,-2.,2.,'',1.,'K/day'] #J/kg.s -> K/day
        plot_dic['RVMTEND_CLUBB']  = ['Humidity Tendencies - CLUBB',1000.*86400, -50., 50.,-20.,20.,'',1.,'g/kg/day']
                                                       
        plot_dic['WPRTP_CLUBB'] = ['w,q - Flux Covariance - CLUBB',1., -0., 600.,-50.,50.,'wq_r',Lv,'W/m^2']
        plot_dic['WPTHLP_CLUBB'] = ['w,thl - Flux Covariance - CLUBB',1., -100., 100.,-50.,50.,'wt_r',cp_air,'W/m^2']
        plot_dic['WPTHVP_CLUBB'] = ['w,thv - Flux Covariance - CLUBB',1., -100., 100.,-50.,50.,'',1.,'W/m^2']
        
        plot_dic['THLP2_CLUBB'] = ['T^2 - Variance - CLUBB',1., 0., 0.15,-50.,50.,'tt_r',1.,'K^2']   
        plot_dic['RTP2_CLUBB'] = ['q^2 - Variance - CLUBB', 1., 0., 2.5,-0.5,0.5,'qq_r',1000.*1000.,'g^2/kg^2'] 
 
        plot_dic['WP2_CLUBB'] = ['w^2 - Variance - CLUBB', 1., 0., 1.5,-0.5,0.5,'ww_r',1.,'m^2/s^2'] 
        plot_dic['WP3_CLUBB'] = ['w^3 - Skewness  - CLUBB',1., 0., 1.5,-0.2,0.2,'www_r',1.,'m^3/s^3']
 
        plot_df = pd.DataFrame.from_dict(plot_dic, orient='index', \
            columns=['long_name','vscale','cmin','cmax','acmin','acmax','var_les','lscale','units'])


    
    
    
    

    
    
    
###############################################
# 2D Mean Height Plotting 
###############################################



    
    if (plot_set=='2d_mean') :
       
        plot_mean_dic['T']      = ['Temperature',1.,260.,305.,-20.,20.,'',1.,'K']
        plot_mean_dic['RELHUM'] = ['Relative Humidity',1.,10., 120.,-100.,100.,'',1.,'%']
        plot_mean_dic['CLOUD']  = ['Cloud Fraction',100., 0., 100.,-80.,80.,'',1.,'%']
        plot_mean_dic['Q']      = ['Specific Humidity',1000., 1., 12.,-5,5.,'q',1000.,'g/kg']

#        plot_df = pd.DataFrame.from_dict(plot_dic, orient='index', \
#            columns=['long_name','vscale','cmin','cmax','acmin','acmax','var_les','lscale','units'])
    
    
### LIST VARIABLE AND INFO IF REQUESTED ###
    
   
    if (all_vars_info):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('++++++++++++ ALL AVAILABLE VARIABLES FOR => ',plot_set,' <= ++++++++++')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        display(plot_df)
#    print(plot2d_df.style.set_table_styles([{'selector':'','props':[('border','4px solid #7a7')]}]))


    return plot_df