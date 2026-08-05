'''
    ###################### KE SIMULATIONS WITH atmospheric_physics EXTERNAL REPO ###################
'''



def scam_cases_grab():



#    iop_case = 'togaII'
#    iop_case = 'gateIII' 
    iop_case = 'arm95' 
#    iop_case = 'arm97'


# ### CAM7 KE cases ###

#    fig_pref = 'zm_cam7_ke_nosmooth'


#    cases = ['FSCAM.T42_T42.XXX.atmos_phys.000','FSCAM.T42_T42.XXX.atmos_phys.001','FSCAM.T42_T42.XXX.atmos_phys.002','FSCAM.T42_T42.XXX.atmos_phys.003','FSCAM.T42_T42.XXX.atmos_phys.004']
#    cnames =  ['CAM7 (default)','CAM7 (KE default)','CAM7 (KE pe2ke_eff=0.2)','CAM7 (KE pini_ke=20)','CAM7 (KE def.+dmpdz=-1.e4)']

    



# ## CAM7 ZM PBL CASES ###

#    fig_pref = 'zm_cam7_pbl_sens_noise'
#   fig_pref = 'zm_cam7_pbl_sens_noise_smthd'
#    fig_pref = 'zmpbl_paperx_c6'
#    fig_pref = 'zmpbl_paperx_c7'
    fig_pref = 'zmpbl_paperx_c7inic'
    
#    fig_pref = 'cam7_cmt_bug'
    
#    fig_pref = 'zm_cam7_ke_tune'
#    

# ZM_PBL Paper Simulations

    cases = ['FSCAM.T42_T42.XXX.cam7-phys.000.L32','FSCAM.T42_T42.XXX.cam7-phys.clm_inic.000.L32']
    cnames =  ['CAM6-L32','CAM6-L32inic']

    
#    cases = ['FSCAM.T42_T42.XXX.cam6-phys.000.L32','FSCAM.T42_T42.XXX.cam6-phys.000.L48','FSCAM.T42_T42.XXX.cam6-phys.000.L58','FSCAM.T42_T42.XXX.cam6-phys.000.L256']
#    cnames =  ['CAM6-L32','CAM6-L48','CAM6-L58','CAM6-L256']

#    cases = ['FSCAM.T42_T42.XXX.cam7-phys.001.L32','FSCAM.T42_T42.XXX.cam7-phys.001.L48','FSCAM.T42_T42.XXX.cam7-phys.001.L58','FSCAM.T42_T42.XXX.cam7-phys.001.L256']
#    cnames = ['CAM7-L32','CAM7-L48','CAM7-L58','CAM7-L256']
    
#    cases = ['FSCAM.T42_T42.XXX.cam7-phys.001.L32','FSCAM.T42_T42.XXX.cam7-phys.001.L48','FSCAM.T42_T42.XXX.cam7-phys.001.L58','FSCAM.T42_T42.XXX.cam7-phys.001.L256']
#    cnames =  ['CAM7-L32','CAM7-L48','CAM7-L58','CAM7-L256']


    


#    cases = ['FSCAM.T42_T42.XXX.cam7_zmpbl.000','FSCAM.T42_T42.XXX.cam7_zmpbl.001','FSCAM.T42_T42.XXX.cam7_zmpbl.002','FSCAM.T42_T42.XXX.cam7_zmpbl.003','FSCAM.T42_T42.XXX.cam7_zmpbl.005','FSCAM.T42_T42.XXX.cam7_zmpbl.006']
#    cnames =  ['CAM7 (default)','CAM7 (ZM PBL Off)','CAM7 (ZM PBL phscale=0.25)','CAM7 (default+1.5K-tiedke_add)','CAM7 (default+all Buoy)','CAM7 (def.+all B+t_add_lcl)']

#    cases = ['FSCAM.T42_T42.XXX.cam7_zmpbl.000','FSCAM.T42_T42.XXX.cam7_zmpbl.001','FSCAM.T42_T42.XXX.cam7_zmpbl.003','FSCAM.T42_T42.XXX.cam7_zmpbl.005','FSCAM.T42_T42.XXX.cam7_zmpbl.007']
#    cnames =  ['CAM7 (default)','CAM7 (ZM PBL Off)','CAM7 (default+1.5K-tiedke_add)','CAM7 (default+all Buoy)','CAM7 (def.+all B+t_add_lcl=1.5K)']

#    cases = ['FSCAM.T42_T42.XXX.cam7_zmall.001','FSCAM.T42_T42.XXX.cam7_zmall.002','FSCAM.T42_T42.XXX.cam7_zmall.003','FSCAM.T42_T42.XXX.cam7_zmall.004','FSCAM.T42_T42.XXX.cam7_zmall.005']
#    cnames =  ['CAM7','+KE','+KE+TKE','+Low_Ent','+ParT']

#    cases = ['FSCAM.T42_T42.XXX.cam7_zmall.001','FSCAM.T42_T42.XXX.cam6_4_032.000']
#    cnames =  ['CAM7 (w/ KE tag)','CAM7 (w/ 032 tag)']

#    cases = ['f.e30.FHIST.SCAM7.XXX.zmpbl.L32.000','f.e30.FHIST.SCAM7.XXX.zmpbl.L32.000ic2']
#    cnames =  ['CAM7-L32-ic1','CAM7-L32-ic2']

#    cases = ['FSCAM.T42_T42.XXX.cam6_4_032.000.L32','FSCAM.T42_T42.XXX.cam6_4_032.000.L48','FSCAM.T42_T42.XXX.cam6_4_032.000.L58','FSCAM.T42_T42.XXX.cam6_4_032.000.L256']
#    cnames =  ['CAM6-L32','CAM6-L48','CAM6-L58','CAM6-L256']


    
    


# ------------------------

    

    cases = [item.replace('XXX', iop_case) for item in cases]

 
    
    
    return cases, cnames, fig_pref

