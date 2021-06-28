"""
    SCAM FORCINGS: IOPs specifics and Libraries 
"""


def scam_iop_settings(iop_case,dir_main):
	
    """
        SCAM FORCINGS IOP CASE SPACE SETTINGS
        
        SAS, PERDIAGO (to do), RICO (to do)

    """
    
    dir_root = '/Users/rneale/Documents/NCAR/PBL/PBL_py_data/iop/'
    
    case_info = {}
    
    
    
    
    if iop_case=='SAS':
         
        case_info['iop_title'] = 'Southeast Atmosphere Study (SipAS) campaign: Ideal day for the Mixed Layer Model (MXLCH, it can be accessed at https://github.com/classmodel/mxlch)'
        case_info['iop_doi'] = 'https://doi.org/10.5194/acp-16-7725-2016'
        
    #### IOP file info. #####
        case_info['iop_file_in'] = dir_root+'ARM95_4scam_c180703'   # Input template
        case_info['iop_file_out'] = dir_root+'SAS_ideal_4scam_22test.no_lsf' # Output forcing file

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

   

        case_info['vname'] = ('pblh','w_sub','shflx','lhflx','eratio','the_bl','the_trop','the_lr','the_adv','q_bl','q_trop','q_lr','q_adv')
        #vval =  (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   6.4e-4, 16.8,    12.8,   -0.004,   1.5e-4) # paper values
        #vval =   (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   0., 11.8,    7.8,   -0.004,   0.) # Drop q, by 5 g/kg, Reinv. SAS specification
        case_info['vval'] =   (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   0., 11.8,    7.8,   -0.004,   0.) # Strict NCAR-PBL SAS specs.
        #vval =   (500,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   0., 16.8,    12.8,   -0.004,   0.) # current values

   


##################################################################################################################################      ################################################################################################################################## 




    if iop_case=='PERDIGAO':
        
        
        case_info['iop_title'] = 'PERDIAGO'
        case_info['iop_doi'] = 'https://doi.org/10.5194/acp-16-7725-2016'
        
    #### IOP file info. #####
        case_info['iop_file_in'] = dir_root+'ARM95_4scam_c180703'   # Input template
        case_info['iop_file_out'] = dir_root+'SAS_ideal_4scam_22test.no_lsf' # Output forcing file

        ### IOP location ###
        case_info['iop_lat'] =  32.5  # Lat location (SAS average of two smapling sites)
        case_info['iop_lon'] = -87.15 # Lon

        case_info['bdate'] = 20170501

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


        case_info['vname'] = ('pblh','w_sub','shflx','lhflx','eratio','the_bl','the_trop','the_lr','the_adv','q_bl','q_trop','q_lr','q_adv')
        case_info['vval'] =   (1000,    9.e-6,   0.1,   0.15,    0.2,     296.6,   298.1,     0.003,   0., 11.8,    7.8,   -0.004,   0.) 
        
        
        
##################################################################################################################################  
##################################################################################################################################  
        
        
        
        
        
        
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
       
        
        
        
        
        
        
### PRINT VALUES FOR THIS IOP ###
        
    print(' ++++++++++++++++++++++ ')   
    print(' ++++ ',iop_case,' ++++ ')   
    print(' ++++++++++++++++++++++ ')   
    
    print('NAME\t\tValue')
    for name, value in case_info.items():
        print('{}\t\t{}'.format(name, value))

        
        
    ############### RETURN VALUES  ###############
    
    return case_info








#################################
#   Chemical Initial Conditions #
#################################


def chem_ics(): 
    
    chem_spec = {}
    chem_spec['O3'] = ['Ozone',25.17,0.007437,26.26,0.004336,23.33,0.007413,28.43,0.004745]  
    chem_sepc['H2O2'] = ['H2O2',1.471,0.0001219,1.475,0.0001114,2.249,0.0007014,0.2173,0.0003611]
    chem_spec['CO'] = ['Carbon Monoxide',107.05,0.006602,108.17,0.009788,107.47,0.009051,85.97,0.002193]
#    chem_spec['NO'] =

    
