
import metpy.constants as mconst


def scam_vars_grab():

    ## Constants ##
    cp_air = mconst.dry_air_spec_heat_press.magnitude # Specific heat for dry air
    grav = mconst.earth_gravity.magnitude       # Gravity ave.
    Lv = mconst.water_heat_vaporization.magnitude       # Latent heat of vaporization
    
    ovar1d = None ; ovar2d = None # If it does not stay None the plot obs
    vscale = 1 ; ovscale1d = 1. ; ovscale2d = 2.
    
    
    '''
        Variables 
    '''
    
    ## Derived varibale set
    vderived = ['MSE'] # Derived variables that have to be calculated
    
    
    
    # Comopsiting 1D and 2D variables
    
    cnamex = 'PRECT' ; xbins=20
    cnamey = 'ZMDT' ; 
    
    
    
    
    
    '''
    
    ###### 2D fields ######
    
    '''
    
    
    ## Variable to be plotted.
    
#    vname = 'RELHUM'   ; vscale = 1. ; units2d='%' ; cmin = 50 ; cmax = 120. ; cmap = 'YlGnBu'
 #   vname = 'Q'   ; vscale = 1000. ; units2d='g/kg' ; cmin = 0.001 ; cmax = 30. ; cmap = 'YlGnBu' ; ovar2d = 'q' ;ovscale2d = 1000. 
#    vname = 'Q'   ; vscale = 1000. ; units2d='g/kg' ; cmin = 0.5 ; cmax = 1.5 ; cmap = 'YlGnBu' 
    
    #vname = 'MSE'   ; vscale = 1.e-6 ; units2d='10^6 J/kg' ; cmin = 0.34 ; cmax = 0.4 ; cmap = 'YlGnBu'
    
#    vname = 'T'   ; vscale = 1.  ; units2d='K' ; cmin = 180 ; cmax = 300. ; cmap = 'RdBu_r'; ovar2d = 'T' ; ovscale2d = 1.
#    vname = 'BUOY' ; vscale = 1. ; units2d='K' ; cmin = -5 ; cmax = 5. ; cmap = 'RdBu_r'
#    vname = 'CLOUD' ; vscale = 100. ; units2d='%' ; cmin = 0 ; cmax = 100. ; cmap = 'Purples'
#    vname = 'CONCLD' ; vscale = 100. ; units2d='%' ; cmin = 0 ; cmax = 50. ; cmap = 'Purples'
    #vname = 'CLDICE'   ; vscale = 1000.
    #vname = 'CLDLIQ' ; vscale = 1000.
    #vname = 'CLDLIQZM' ; vscale = 1000.
    #vname = 'DLFZM' ; vscale = 1000.
    #vname = 'QRS' ; vscale = 86400.
    #vname = 'QRL'   ; vscale = 86400. ; units2d='K/day' ; cmin = -10 ; cmax = 10. ; cmap = 'RdBu_r'
    
#    vname = 'ZMDQ'   ; vscale = 1000.*86400. ; units2d='g/kg/day' ; cmin = -6 ; cmax = 6. ; cmap = 'RdBu'
#    vname = 'ZMDQ'   ; vscale = 1000.*86400. ; units2d='g/kg/day' ; cmin = 0.2 ; cmax = 1.8 ; cmap = 'RdBu'
    
    vname = 'ZMDT'   ; vscale = 86400. ; units2d='K/day' ; cmin = -5 ; cmax = 10. ; cmap = 'RdBu_r'
#    #vname = 'ZMDLF'   ; vscale = 1000.*86400. ; units2d='K/day' ; cmin = -10 ; cmax = 10. ; cmap = 'RdBu_r'
#    vname = 'DMPDZ'   ; vscale = 1000. ; units2d='/km' ; cmin = -1.05; cmax = -0.05 ; cmap = 'RdBu_r'
    
    
#    vname = 'STEND_CLUBB'   ; vscale = 86400./cp_air ; units2d='K/day' ; cmin = -15 ; cmax = 15. ; cmap = 'RdBu_r'
#    vname = 'MPDT'   ; vscale = 86400./cp_air ; units2d='K/day' ; cmin = -15 ; cmax = 15. ; cmap = 'RdBu_r'


#    vname = 'TKE'   ; vscale = 1.; units2d='J/kg' ; cmin = 0. ; cmax = 2. ; cmap = 'Purples'
    #vname = 'RVMTEND_CLUBB'   ; vscale = 86400*1000. ; units2d='g/kg/day' ; cmin = -5. ; cmax = 5. ; cmap = 'RdBu_r'
#    vname = 'WP2_CLUBB'   ; vscale = 1.; units2d='m2/s2' ; cmin = 0. ; cmax = 1. ; cmap = 'Purples'
#    vname = 'WP3_CLUBB'   ; vscale = 1.; units2d='m3/s3' ; cmin = -5. ; cmax = 5. ; cmap = 'RdBu_r'
    #vname = 'UP2_CLUBB'   ; vscale = 1.; units2d='m2/s2' ; cmin = 0. ; cmax = 5dir_c

    
    
    
#    vname = 'DCQ'   ; vscale = 1000.*86400. ; units2d='g/kg/day' ; cmin = -8 ; cmax = 8. ; cmap = 'RdBu_r'
#    vname = 'DTCOND' ; vscale = 86400. ; units2d='K/day' ; cmin = -15 ; cmax = 15. ; cmap = 'RdBu_r'
    
#    vname = 'ZMMU' ; vscale = 3600. ; units2d='kg/m^2/hr' ; cmin = -10. ; cmax = 50. ; cmap = 'PuBuGn'
#    vname = 'ZMMD' ; vscale = 3600. ; units2d='kg/m^2/hr' ; cmin = -8 ; cmax = 4. ; cmap = 'PuBuGn_r'
#    vname = 'WINCLD' ; vscale = 1. ; units2d='m/s' ; cmin = -20; cmax = 20; cmap = 'RdBu_r'
#    vname = 'KEPAR'  ; vscale = 1. ; units2d='W/m2' ; cmin = -20; cmax = 80; cmap = 'RdBu_r'


    
    
    '''
        ###### 1D fields ######    
    '''
    
#    vname1d = 'SHFLX' ; vscale1d = 1. ; units1d='W/m^2' ; pmin = 0. ; pmax= 20.
#    vname1d = 'LHFLX' ; vscale1d = 1. ;  units1d='W/m^2' ; pmin = 0. ; pmax= 300.  ; ovar1d = 'lhflx' ; ovscale1d = 1. 
    #vname1d = 'U10' ; vscale1d = 1. ;  units1d='m/s' ; pmin = 0. ; pmax= 10.
    #vname1d = 'TAUX' ; vscale1d = 1. ;  units1d='kg/m^2/s' ; pmin = -0.2 ; pmax= 0.0
    #vname1d = 'ZBOT' ; vscale1d = 1. ;  units1d='m' ; pmin = 0. ; pmax= 150.
#    vname1d = 'PRECT' ; vscale1d = 86400.*1000. ;  units1d='mm/day' ; pmin = 0.01 ; pmax= 60. ; ovar1d = 'Prec' ; ovscale1d = 86400. 
     
    
#    vname1d = 'PRECZ' ; vscale1d = 86400.*1000. ;  units1d='mm/day' ; pmin = 0. ; pmax= 45.
    vname1d = 'PRECC' ; vscale1d = 86400.*1000. ;  units1d='mm/day' ; pmin = 0. ; pmax= 45.
#    vname1d = 'PRECL' ; vscale1d = 86400.*1000. ;  units1d='mm/day' ; pmin = 0. ; pmax= 45.
#    vname1d = 'PRECT' ; vscale1d = 1. ;  units1d='J/kg' ; pmin = 0. ; pmax= 1000.
#    vname1d = 'CAPE' ; vscale1d = 1. ;  units1d='J/kg' ; pmin = 0. ; pmax= 1000.    
#    vname1d = 'TLCL' ; vscale1d = 1. ;  units1d='K' ; pmin = 280. ; pmax= 300.
#    vname1d = 'PLCL' ; vscale1d = 1. ;  units1d='mb' ; pmin = 800. ; pmax= 1000.
#    vname1d = 'LEL' ; vscale1d = 1. ;  units1d='mb' ; pmin = 50. ; pmax= 1000.
#    vname1d = 'KHMAX' ; vscale1d = 1. ;  units1d='level' ; pmin = 850. ; pmax= 1000.
#    vname1d = 'PKHMAX' ; vscale1d = 1. ;  units1d='pressure' ; pmin = 800. ; pmax= 1050. # Pressure location of KHMAX
#    vname1d = 'PBLH' ; vscale1d = 1. ;  units1d='meters' ; pmin = 50. ; pmax= 4000. # Pressure location of KHMAX
    #vname1d = 'TKE&IC' ; vscale1d = 1. ;  units1d='meters' ; pmin = 0. ; pmax= 40. # Pressure location of KHMAX
    #vname1d = 'LWCF' ; vscale1d = 1. ;  units1d='W/m^2' ; pmin = 0. ; pmax = 100.
    
    #vname1d = 'CLDLOW' ; vscale1d = 1.
    
    return vname, vscale, units2d, cmin, cmax, cmap, ovar2d, ovscale2d, vname1d, vscale1d, units1d, pmin, pmax, ovar1d, ovscale1d     



