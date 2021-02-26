"""
    SCAM FORCINGS: IOPs specifics and Libraries 
"""


def scam_iop_settings(iop_case,dir_main):
	
    """
        SCAM FORCINGS IOP CASE SPACE SETTINGS
        
        SAS, PERDIAGO (to do), RICO (to do)

    """
    if iop_case=='SAS':
    
        case_info = {}

    #### IOP file info. #####
        case_info['iop_file_in'] = dir_main+'iop/ARM95_4scam_c180703.nc'   # Input template
        case_info['iop_file_out'] = dir_main+'iop/SAS_ideal_4scam_22.no_lsf.nc' # Output forcing file

        ### IOP location ###
        case_info['iop_lat'] =  32.5  # Lat location (SAS average of two smapling sites)
        case_info['iop_lon'] = -87.15 # Lon

        case_info['bdate'] = 20130610

        case_info['zoffset'] = 6. # Increment to reach Z time 


        ## Time info (strict SAS configuration) ##
        case_info['dtime'] = 2*60.    # Time interval (minutes*60=sec) for each time point on file (does not have to be SCAN's DTIME)

        ## Surface flux time profile
        case_info['lflux_tstart'] = 7.  
        case_info['lflux_tend'] = 19.5      # Start/end times LH

        case_info['sflux_tstart'] = 6.   
        case_info['sflux_tend'] = 19.5      # SH

        case_info['loc_tstart'] = 5.
        case_info['loc_tend'] = 18.


        case_info['ps'] = 96700. # psurf [pa] - On the file ps = 967mb

        ### Inversation Layer Info. ###

        case_info['zinv_bot'] = 352.5 # Inversation height bottom.
        case_info['zinv_top'] = 442.5 # Inversation height top.

        ### Wind (geostrophic) ###


        case_info['u_g'] = 2.
        case_info['v_g'] = 0.

    #p0 = 100000. # pref [pa]

    #### Specific CASE VARIABLE SETTIGS #####

        case_info['vdesc'] = ('Initial BL height', \
             'Subsidence rate', \
             'Surface sensible heat flux',\
             'Surface latent heat flux',\
             'Entrainment/surface heat flux ratio'\
             'Initial BL potential temperature',\
             'Initial FT potential temperature',\
             'Potential temperature lapse rate FT',\
             'Advection of potential temperature',\
             'Initial BL specific humidity',\
             'Initial FT specific humidity',\
             'Specific humidity lapse rate FT',\
             'Advection of specific humidity')


        case_info['vname'] = ('pblh','w_sub','shflx','lhflx','eratio','the_bl','the_trop','the_lr','the_adv','q_bl','q_trop','q_lr','q_adv')
        #vval =  (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   6.4e-4, 16.8,    12.8,   -0.004,   1.5e-4) # paper values
        #vval =   (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   0., 11.8,    7.8,   -0.004,   0.) # Drop q, by 5 g/kg, Reinv. SAS specification
        case_info['vval'] =   (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   0., 11.8,    7.8,   -0.004,   0.) # Strict NCAR-PBL SAS specs.
        #vval =   (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   0., 16.8,    12.8,   -0.004,   0.) # current values

        
        
### PRINT VALUES FOR THIS IOP ###
        
    print(' ++++++++++++++++++++++ ')   
    print(' ++++ ',iop_case,' ++++ ')   
    print(' ++++++++++++++++++++++ ')   
    
    print('NAME\t\tValue')
    for name, value in case_info.items():
        print('{}\t\t{}'.format(name, value))

        
        
    ############### RETURN VALUES  ###############
    
    return case_info