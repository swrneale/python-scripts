'''
    REPOSITORY OF CASE AND VARIABLE CHOICES

'''



def get_case_all(case_group):

    
    '''
        ################### VARIABLE TO PLOT ################
    '''

    obs_cases = []
    
    
    
    ## 2D variables

#    var = 'RESTOM' ; obs_cases = ['']
    
#    var = 'PRECC' ; obs_cases = ['']
    #var = 'FREQZM' ; obs_cases = ['']
    #var = 'PRECL' ; obs_cases = ['']
#    var = 'PRECT' ; obs_cases = ['TRMM'] # TRMM,GPCP
    #var = 'CLDLOW' ; obs_cases = []
    var = 'SWCF' ; obs_cases = [''] # CERES-EBAF
#    var = 'LWCF' ; obs_cases = ['CERES-EBAF']
#    var = 'FSNT' ; obs_cases = ['CERES-EBAF']
#    var = 'FLNT' ; obs_cases = ['CERES-EBAF']
    
#    var = 'LHFLX' ; obs_cases = [''] # LARYEA,WHOI,ERAI
    #var = 'SHFLX' ; obs_cases = ['LARYEA']
#    var = 'TMQ' ; obs_cases = ['NVAP']
#    var = 'TPERT_ZM' ; obs_cases = ['']
    
    
#    var = 'PCONVT' ; obs_cases = []
#    var = 'TAUX' ; obs_cases = ['']
    #var = 'TAUY' ; obs_cases = ['MERRA']
    # 3D variables
    
    #var = 'Q' ; obs_cases = ['MERRA']
#    var = 'TS' ; obs_cases = ['HadISST_PI']
    
    
    
    
    #var = 'SFbc_a4' ; obs_cases = []
    #var = 'SFpom_a4' 
    #var = 'SFso4_a1' 
    #var = 'pom_a4_CLXF' 
    #var = 'pom_a4' ; vproc = 'pvint'
    
    
    



    '''
        ################# CASE SELECTION ############
    '''

    if case_group == 1:
    
        fig_pref = 'CAM7_4K_315_316_b'

        dcases = {'f316' : 'f.e30_alpha08b.FHISTC_MTso.ne30_t232_wgx3.316',
                 'f315' :   'f.e30_alpha08b.FHISTC_MTso.ne30_t232_wgx3.315',
                 'f315 4K' : 'f.e30_alpha08b.FHISTC_MTso.ne30_t232_wgx3.315_SST4K',
                 'f316 4K' : 'f.e30_alpha08b.FHISTC_MTso.ne30_t232_wgx3.316_SST4K',
        }

        
    
 #       dcases = {'CAM7-032'          : 'f.cam6_4_032.FLTHIST_ne30.cam7_dev.002',
 #                 'CAM7-089'          : 'f.cam6_4_089.FLTHIST_ne30.cam7.001',
 #                 '0.75*lhf'          : 'f.e30_cam6_4_078.FHISTC_LTso.ne30.lhf_roughphis_SGH_grlnd28k.001',
 #                 'qsat (T2m)'         : 'f.e30_cam6_4_078.FHISTC_LTso.ne30.ssq_T2m_roughphis_SGH_grlnd28k.001',
 #                 'flux all (T2m)'     : 'f.e30_cam6_4_078.FHISTC_LTso.ne30.flux_all_T2m_roughphis_SGH_grlnd28k.001',
 #                 'flux all (T2m:SST)' :'f.e30_cam6_4_078.FHISTC_LTso.ne30.flux_all_Tsave_roughphis_SGH_grlnd28k.001',
 #                 'UA'                 :'f.e30_cam6_4_078.FHISTC_LTso.ne30.ocnflx2_roughphis_SGH_grlnd28k.001',
 #                 'G&G qsat'           :'f.e30_cam6_4_078.FHISTC_LTso.ne30.baseline156_qsat.001'}
    
    
    
    #    dcases = {'089 bline'    : 'f.cam6_4_089.FLTHIST_ne30.cam7.001',
    #              'dmpdz.001a'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.001a',
    #              'dmpdz.005b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.005b',
    #              'dmpdz.089'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_dmpdz.002',
    #              'zmke.001'     : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_ke.001',
    #              'zmke.002b'     : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.002b',
    #              'zmke.003b'     : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.003b',
    #              'pblpar.001'     : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_pblpar.001',
    #              'pblpar.002'     : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_pblpar.002'
    #             }             
    
    #    dcases = {'CAM7-089'    : 'f.cam6_4_089.FLTHIST_ne30.cam7.001',
    #              'dmpdz-5k'     :'f.cam6_4_089.FLTHIST_ne30.cam7_zm_dmpdz.002',
    #              'dmpdz-3k'     :'f.cam6_4_089.FLTHIST_ne30.cam7_zm_dmpdz.003',   
    #              'KE-par'       : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_ke.001',
    #              'zmpar-1.0pblh'      : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_pblpar.001',
    #              'zmpar-0.25pblh'     : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_pblpar.002'
    #             }    
    
    
    #  dcases = {'dmpdz.01a'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.001a',
    #              'dmpdz.04b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.004b',
    #              'dmpdz.05b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.005b',
    #              'dmpdz.089'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_dmpdz.002',
    #              'dmpdz.07b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.007b',
    #              'dmpdz.08b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.008b',
    #              'dmpdz.09b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.009b',
    #              'dmpdz.10b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.010b',
    #              'dmpdz.11b'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.011b'}
    
    #    fig_pref = 'CAM7-ZM-sens'
    
    #    dcases = {'032'      : 'f.cam6_4_032.FLTHIST_ne30.cam7_dev.002',
    #              '089'      : 'f.cam6_4_089.FLTHIST_ne30.cam7.001',
    #              '089dmpdz'      : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_dmpdz.001',
    #              '0.5xtau'  : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tau_x0.5.001',
    #              '2xtau'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tau_x2.001',
    #              'cmt_0.4'  : 'f.cam6_4_032.FLTHIST_ne30.cam7_dev.005',
    #              'cmt_0.1'  : 'f.cam6_4_032.FLTHIST_ne30.cam7_dev.006',
    #              'tiedke 0K' : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke0K.001',
    #              'tiedke 2K'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke2K.001',
    #              'c0_lnd_ocn'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_c0_lnd.001', }
    
#        fig_pref = 'CAM7-ZM-sens_subset_epac'
    
#        dcases = {
#                  'CAM7-089 (0.5K)'      : 'f.cam6_4_089.FLTHIST_ne30.cam7.001',
#                  'tiedke 0K'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke0K.001',
#                  'tiedke 1K'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke1K.001',
#                  'tiedke 1.5K'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke1.5K.001',
#                  'tiedke 2K' : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke2K.001',
#                  'tiedke 3K'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke3K.001',}
        
#        fig_pref = 'CAM7-ZM-sens_subset3'
        
#        dcases = {
#                  'CAM7-089'      : 'f.cam6_4_089.FLTHIST_ne30.cam7.001',
#                  'tiedke 0K' : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke0K.001',
#                  'tiedke 1K'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke1K.001',
#                  'cmt 0.4' : 'f.cam6_4_032.FLTHIST_ne30.cam7_dev.005',
#                  'cmt 0.1' : 'f.cam6_4_032.FLTHIST_ne30.cam7_dev.006',
#                  'KE def.' : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.002b',
#                  'KE TKE.' : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.003b',
#                  'KE def2..' : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.008b',
#                  'tscool=-30' :'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tscool.001',
#                  'dmpdz 5k'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_dmpdz.002', 
#                  'dmpdz x2 low 3k'    : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.007b',
#                }

 #       fig_pref = 'AMWG-tpert_tiedke' 
#        fig_pref = 'CAM7-nogust' 
        
#        dcases = {
#                 'CAM6'                           :'f.e20.FHIST.f09_f09.cesm2_1.001',
#                 'CAM6'                           : 'f.e21.FHIST.f09_f09.cam6_sfc_pod.002',
#                 'CAM7-ctrl'                      : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.001',
#                 'CAM7-ctrl'                      : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.001',
#                  '5K,tpert,sgh30max=100'         : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.002',
#                  '3K,tpert,sgh30max=100'         : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.003',
#                  '1K,tpert,sgh30max=100'         : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.004',
#                  '5K,tpert,sgh30max=200'         : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.005',
#                  '3K,tpert,fixed.'               : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.006',
#                  '3K,tiedke\_add,fixed (orig)'   : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke3K.001', 
#                  '2K,tiedke\_add,fixed (orig)'   : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke2K.001',            
#                  '3K,tpert,sgh30max=0'           : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.007',
#                  't-add=3K'                      : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.008',
#                  't-add=3K,no b\'(T)'            : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.009',
#                  '3K,tpert->t-add,sgh30max=100'  : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.010',
#                  '3K,tpert,zmax=4000'            : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.011',
#                  '3K,tpert->t-add,zmax=4000'     : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.012',
#                  '6K,tpert->t-add,zmax=4000'     : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.013',
#                  '3K,tpert->t-add,land only'     : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.014',
#                  '3K,tpert->t-add,land+zmax=4000'     : 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.015',

#        } 
            
    if case_group == 2:
    
    #    fig_pref = 'CAM7_to_CESM3_bias_epac'
        
    #    dcases = {'Uncoupled Bias' : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.001a',
    #             'Uncoupled Bias (with change)' : 'f.cam6_4_032.FLTHIST_ne30.cam7_ke.002b',
    #             'Coupled Biasx': 'b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192', 
    #              'Coupled Bias': 'b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192', 
    #              'Coupled Bias (with change)': 'b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.193',          
    #             }
    
    #    dcases = {'CESM3 (198)': 'b.e30_alpha07b_dev.B1850C_LTso.ne30_t232_wgx3.198', 
    #              'CAM7 (AMIP)' : 'f.cam6_4_089.FLTHIST_ne30.cam7.001',
    #              'CESM2' : 'b.e20.B1850.f09_g17.pi_control.all.297',
    #              'CESM1' : 'b.e11.B1850C5CN.f09_g16.005'}
    
    #
#        fig_pref = 'CAM7_to_CESM3_bias_epac'
#        fig_pref = 'CAM7_to_CESM3_bias_tiedke'
        
#        dcases = {'234': 'b.e30_alpha07c_cesm.B1850C_LTso.ne30_t232_wgx3.234', 
#                  '249': 'b.e30_alpha07c_cesm.B1850C_LTso.ne30_t232_wgx3.249',
#                  '250':'b.e30_alpha07c_cesm.B1850C_LTso.ne30_t232_wgx3.250',
#                  'CESM2' : 'b.e20.B1850.f09_g17.pi_control.all.297',
#                  'CESM1' : 'b.e11.B1850C5CN.f09_g16.005'}
    
#        fig_pref = 'CAM7_comp_CESM3_bias_tiedke'
        
#        dcases = {'249': 'b.e30_alpha07c_cesm.B1850C_LTso.ne30_t232_wgx3.249', 
#                  '234 0.5K (ctrl)': 'b.e30_alpha07c_cesm.B1850C_LTso.ne30_t232_wgx3.234',
#                  '250 1.5K':'b.e30_alpha07c_cesm.B1850C_LTso.ne30_t232_wgx3.250',
#                  'FHIST 0.5K (ctrl)'      : 'f.e30_cam6_4_127.FHISTC_LTso.ne30.baseline_DGLC.001',
#                  'FHIST 3K'    : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke1.5K.001',
#                 }

        fig_pref = 'CAM7_MCSP'
        
        dcases = {'CAM6': 'f.e21.FHIST.f09_f09.cam6_sfc_pod.002', 
                  'CAM7': 'f.cam6_4_130.FLTHIST_ne30.cam7-tpert.001',
                  '2K,tiedke\_add,fixed (orig)'   : 'f.cam6_4_089.FLTHIST_ne30.cam7_zm_tiedke2K.001', 
                  'CAM7 (MCSP.v2)':'CTL_divdamp_fmt_c64126_MCSP_v2',
                  'CAM7 (MCSP.test1)':'CTL_divdamp_fmt_c64126_MCSP_test1',}

    
    

    
    if case_group == 3:
    
        fig_pref = 'aero_bb_bc_sims'
    
        dir_fig = '/glade/u/home/rneale/python/python-figs/aerosols/'
    
        
        if pregion == 'SAm':
    
            nrow_scale = 3
            ncol_scale = 5
            
            dcases = {   'Control'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.000',                 
                      'Global 0.5x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.002a',
                      'Global 2.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.001a',
                      'Global 10.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.009a',
                      'Amazon 0.5x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.007a',
                      'Amazon 2.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.006a',
                      'Amazon 10.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.008a'}
    
        
    
        
        if pregion == 'Boreal':
    
            nrow_scale = 6
            ncol_scale = 4
            
            dcases = {  'Control'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.000',
                      'Global 0.5x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.002a',
                      'Global 2.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.001a',
                      'Global 10.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.009a',
                      'Boreal 0.5x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.004a',
                      'Boreal 2.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.003a',
                      'Boreal 10.0x'    : 'f.e21.F2000climo.f19_f19.cesm2_1.dfires.005a'}
    
    
    
    if case_group == 4:
    
            fig_pref = 'aero_bb_bc_sims'
        
            nrow_scale = 1
            ncol_scale = 3
            
            dcases = { 'FMTHIST (numcin=3)'    : 'QBOtune_frontbugfix_zmcin3_eff015_fmt_c64116',
                       'FMTHIST (numcin=1)'    : 'CTL_fmt_c64116',
                     } 
    
    










    return dcases,var,obs_cases,fig_pref



