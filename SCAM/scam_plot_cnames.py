'''
    ###################### KE SIMULATIONS WITH atmospheric_physics EXTERNAL REPO ###################
'''



def scam_cases_grab():



#    iop_case = 'togaII'
#    iop_case = 'gateIII' 
    iop_case = 'arm95' 
#    iop_case = 'arm97'

    
#### CAM7 KE cases ###

#    fig_pref = 'zm_cam7_ke_nosmooth'
    

#    cases = ['FSCAM.T42_T42.XXX.atmos_phys.000','FSCAM.T42_T42.XXX.atmos_phys.001','FSCAM.T42_T42.XXX.atmos_phys.002','FSCAM.T42_T42.XXX.atmos_phys.003','FSCAM.T42_T42.XXX.atmos_phys.004']
#    cnames =  ['CAM7 (default)','CAM7 (KE default)','CAM7 (KE pe2ke_eff=0.2)','CAM7 (KE pini_ke=20)','CAM7 (KE def.+dmpdz=-1.e4)']

    


    
### CAM7 ZM PBL CASES ###

#    fig_pref = 'zm_cam7_pbl_sens_noise'
#   fig_pref = 'zm_cam7_pbl_sens_noise_smthd'
    fig_pref = 'zm_cam7_zm_def_check2'
#    fig_pref = 'zm_cam7_ke_tune'
#    




#    cases = ['FSCAM.T42_T42.XXX.cam7_zmpbl.000','FSCAM.T42_T42.XXX.cam7_zmpbl.001','FSCAM.T42_T42.XXX.cam7_zmpbl.002','FSCAM.T42_T42.XXX.cam7_zmpbl.003','FSCAM.T42_T42.XXX.cam7_zmpbl.005','FSCAM.T42_T42.XXX.cam7_zmpbl.006']
#    cnames =  ['CAM7 (default)','CAM7 (ZM PBL Off)','CAM7 (ZM PBL phscale=0.25)','CAM7 (default+1.5K-tiedke_add)','CAM7 (default+all Buoy)','CAM7 (def.+all B+t_add_lcl)']

#    cases = ['FSCAM.T42_T42.XXX.cam7_zmpbl.000','FSCAM.T42_T42.XXX.cam7_zmpbl.001','FSCAM.T42_T42.XXX.cam7_zmpbl.003','FSCAM.T42_T42.XXX.cam7_zmpbl.005','FSCAM.T42_T42.XXX.cam7_zmpbl.007']
#    cnames =  ['CAM7 (default)','CAM7 (ZM PBL Off)','CAM7 (default+1.5K-tiedke_add)','CAM7 (default+all Buoy)','CAM7 (def.+all B+t_add_lcl=1.5K)']

#    cases = ['FSCAM.T42_T42.XXX.cam7_zmall.001','FSCAM.T42_T42.XXX.cam7_zmall.002','FSCAM.T42_T42.XXX.cam7_zmall.003','FSCAM.T42_T42.XXX.cam7_zmall.004','FSCAM.T42_T42.XXX.cam7_zmall.005']
#    cnames =  ['CAM7','+KE','+KE+TKE','+Low_Ent','+ParT']
    
    cases = ['FSCAM.T42_T42.XXX.cam7_zmall.001','FSCAM.T42_T42.XXX.cam6_4_032.000']
    cnames =  ['CAM7 (w/ KE tag)','CAM7 (w/ 032 tag)']
    
    
#    cases  = ['FSCAM.T42_T42.XXX.cam7_zmall.001b','FSCAM.T42_T42.XXX.cam6_4_032.001','FSCAM.T42_T42.XXX.cam6_4_032.000','FSCAM.T42_T42.XXX.cam7_zmpbl.000','FSCAM.T42_T42.XXX.atmos_phys.000']
#    cnames = ['CAM7-032-zmke.001b','CAM7-032-nozmpbl','CAM7-032','CAM7-zmpbl','CAM7-atmos_phys']

    
    
# ------------------------

    

    cases = [item.replace('XXX', iop_case) for item in cases]

    
    
    ################## OTHER CASES #################
    
    #cases = ['FSCAM.T42_T42.togaII.100','FSCAM.T42_T42.togaII.100b','FSCAM.T42_T42.togaII.101','FSCAM.T42_T42.togaII.101b']
    #cases = ['FSCAM.T42_T42.togaII.001','FSCAM.T42_T42.togaII.004','FSCAM.T42_T42.togaII.002','FSCAM.T42_T42.togaII.003','FSCAM.T42_T42.togaII.001.L32','FSCAM.T42_T42.togaII.001.L256']
    #cnames = ['CAM6 (#CIN=1)','CAM6 (#CIN=2)','CAM6 (#CIN=3)','CAM6 (#CIN=5)','CAM6 (L32)','CAM6 (L256)']
    
    #cases = ['FSCAM.T42_T42.togaII.001','FSCAM.T42_T42.togaII.004','FSCAM.T42_T42.togaII.003','FSCAM.T42_T42.togaII.001.L256','FSCAM.T42_T42.togaII.001.L256.nolev1zm','FSCAM.T42_T42.togaII.zm.ke002']
    #cnames = ['CAM6 (#CIN=1)','CAM6 (#CIN=2)','CAM6 (#CIN=5)','CAM6 (L256)','CAM6 (L256-nolev1zm)','CAM6 (tfreez=-10)','FSCAM.T42_T42.togaII.001']
    
    #cases = ['FSCAM.T42_T42.togaII.001','FSCAM.T42_T42.togaII.004','FSCAM.T42_T42.togaII.003','FSCAM.T42_T42.togaII.001.L64','FSCAM.T42_T42.arm97.001','FSCAM.T42_T42.arm97.001.L64']
    #cnames = ['CAM6 (#CIN=1)','CAM6 (#CIN=2)','CAM6 (#CIN=5)','CAM6 (L64,#CIN=1)','CAM6 (ARM97)','CAM6 (ARM97, L64)']
    
    #cases = ['FSCAM.T42_T42.togaII.001','FSCAM.T42_T42.togaII.001.org00']
    #cnames = ['CAM6','CAM6 (zm_org)']
    
    #cases = ['FSCAM.T42_T42.togaII.001','FSCAM.T42_T42.togaII.001.sflux01','FSCAM.T42_T42.togaII.001.L256','FSCAM.T42_T42.togaII.001.L256.sflux01','FSCAM.T42_T42.togaII.001.sflux2']
    #cnames = ['CAM6','CAM6 (5m ref.)','CAM6 (L256)','CAM6 (L256, 5m ref.)','CAM6 (Zheng scheme)']
    
    #cases = ['FSCAM.T42_T42.arm97.zm.ke000','FSCAM.T42_T42.arm97.zm.ke000','FSCAM.T42_T42.arm97.zm.par001']
    #cnames = ['CAM6','CAM6-KE.ZM','CAM6-PBLpar']
    
    #cases = ['FSCAM.T42_T42.togaII.zm.ke000','FSCAM.T42_T42.togaII.zm.ke000.L48','FSCAM.T42_T42.togaII.zm.ke000.L58','FSCAM.T42_T42.togaII.zm.ke010',
    #        'FSCAM.T42_T42.togaII.zm.ke008.L48','FSCAM.T42_T42.togaII.zm.ke009.L48','FSCAM.T42_T42.togaII.zm.ke009.L58','FSCAM.T42_T42.togaII.zm.ke010.L58']
    #cnames = ['L32-ctrl','L48-ctrl','L58-ctrl','L32-KE010','L48-KE008','L48-KE009','L58-KE009','L58-KE010']
    
    
    
    #cases = ['FSCAM.T42_T42.togaII.zm.ke000.L58','FSCAM.T42_T42.togaII.zm.ke010.L58']
    #cnames = ['CAM6 (L58)','KEparcel (L58)']
    
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.001','FSCAM.T42_T42.togaII.cam_ke.001.L48','FSCAM.T42_T42.togaII.cam_ke.002.L48']
    #cnames = ['CAM6 (L58)','KEparcel (L58)','KEparcel (L48)','pdirKEparcel+PBLparcel (L48)']
    
    #cases = ['FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.001','FSCAM.T42_T42.arm97.cam_ke.001.L48']
    #cnames =  ['CAM6 (L58)','KEparcel (L58)','KEparcel (L48)']
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke.001','FSCAM.T42_T42.togaII.cam_ke.001.L48','FSCAM.T42_T42.togaII.cam_ke.001.L32']
    #cnames =  ['KEparcel (L58)','KEparcel (L48)','KEparcel (L32)']
    
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L48','FSCAM.T42_T42.togaII.cam_ke.000.L32','FSCAM.T42_T42.togaII.001.L256']
    #cnames =  ['L58','L48','L32','L256']
    
    
    
    
    
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L48','FSCAM.T42_T42.togaII.cam_ke.000.L32','FSCAM.T42_T42.togaII.cam_ke.001','FSCAM.T42_T42.togaII.cam_ke.001.L48','FSCAM.T42_T42.togaII.cam_ke.001.L32']
    #cases = ['FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.000.L48','FSCAM.T42_T42.arm97.cam_ke.000.L32','FSCAM.T42_T42.arm97.cam_ke.001','FSCAM.T42_T42.arm97.cam_ke.001.L48','FSCAM.T42_T42.arm97.cam_ke.001.L32']
    
    
    #cnames =  ['L58','L48','L32','KEparcel (L58)','KEparcel (L48)','KEparcel (L32)']
    
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.002','FSCAM.T42_T42.togaII.cam_ke.003','FSCAM.T42_T42.togaII.cam_ke.004']
    
    #cases = ['FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.002','FSCAM.T42_T42.arm97.cam_ke.003','FSCAM.T42_T42.arm97.cam_ke.004']
    #cnames =  ['L58','KEparcel (L58)','PBL Parcel (L58)','KE/PBL Parcel (L58)']
    
    #cases = ['FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.002','FSCAM.T42_T42.arm97.cam_ke.003','FSCAM.T42_T42.arm97.cam_ke.004']
    #cnames =  ['L58','KEparcel (L58)','PBL Parcel (L58)','KE/PBL Parcel (L58)']
    
    
    
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.001']
    #cnames =  ['CAM6','CAM6-ZMKEpbl']
    
    
    #cases = ['FSCAM.T42_T42.arm97.cam_ke.003','FSCAM.T42_T42.arm97.cam_ke_cam6dev.000']
    #cnames =  ['CAM6+ZMpbl','CAM6-dev']
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke_cam6dev.000','FSCAM.T42_T42.togaII.cam_ke.003','FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L32','cam5.togaII','cam4.togaII','cam3.togaII']
    #cases = ['FSCAM.T42_T42.arm95.cam_ke_cam6dev.000','FSCAM.T42_T42.arm95.cam_ke.003','FSCAM.T42_T42.arm95.cam_ke.000','FSCAM.T42_T42.arm95.cam_ke.000.L32','cam5.arm95','cam4.arm95','cam3.arm95']
    #cnames = ['CAM6-dev','CAM6-L58-PBLpar','CAM6-L58','CAM6','CAM5','CAM4','CAM3']
    
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke_cam6dev.000','FSCAM.T42_T42.togaII.cam_ke.003','FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L32']
    #cases = ['FSCAM.T42_T42.arm95.cam_ke_cam6dev.000','FSCAM.T42_T42.arm95.cam_ke.003','FSCAM.T42_T42.arm95.cam_ke.000','FSCAM.T42_T42.arm95.cam_ke.000.L32']
    #cases = ['FSCAM.T42_T42.arm97.cam_ke_cam6dev.000','FSCAM.T42_T42.arm97.cam_ke.003','FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.000.L32']
    #cnames = ['CAM6-L58-dev','CAM6-L58-PBLpar','CAM6-L58','CAM6']
    
    
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke_cam6dev.000','FSCAM.T42_T42.togaII.cam_ke.003','FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L32','FSCAM.T42_T42.togaII.cam6dev_zm.000','FSCAM.T42_T42.togaII.cam6dev_zm.001']
    #cnames = ['CAMdev','CAM6-L58-PBLpar','CAM6-L58','CAM6','CAMdev-L58-0.2dmpdz','CAMdev-L58-numcin2']
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke_cam6dev.000','FSCAM.T42_T42.togaII.cam6dev_zm.000','FSCAM.T42_T42.togaII.cam6dev_zm.003','FSCAM.T42_T42.togaII.cam6dev_zm.004','FSCAM.T42_T42.togaII.cam6dev_zm.001']
    #cases = ['FSCAM.T42_T42.arm95.cam_ke_cam6dev.000','FSCAM.T42_T42.arm95.cam6dev_zm.000','FSCAM.T42_T42.arm95.cam6dev_zm.003','FSCAM.T42_T42.arm95.cam6dev_zm.004','FSCAM.T42_T42.arm95.cam6dev_zm.001']
    #cnames = ['CAMdev','CAMdev-0.2dmpdz','CAMdev-numcin2','CAMdev-numcin3','CAMmydev-numcin2']
    
    #cases = ['FSCAM.T42_T42.togaII.cam_ke_cam6dev.000','FSCAM.T42_T42.togaII.cam6dev_zm.003','FSCAM.T42_T42.togaII.cam6dev_zm.001','FSCAM.T42_T42.togaII.cam6dev_zm.002']
    #cnames = ['CAMdev','CAMdev-L58-numcin2','CAMdev-L58-numcin3']
    
    #cases = ['FSCAM.T42_T42.arm95.cam_ke_cam6dev.000','FSCAM.T42_T42.arm95.cam6dev_zm.000','FSCAM.T42_T42.arm95.cam6dev_zm.003','FSCAM.T42_T42.arm95.cam6dev_zm.002']
    #cnames = ['CAMdev','CAMdev-L58-0.2dmpdz','CAMdev-L58-numcin2','CAMdev-L58-numcin3']
    
    
    #cases = ['FSCAM.T42_T42.rico.000','FSCAM.T42_T42.rico.001']
    #cnames = ['CAM6-000','CAM6-001']

    return cases, cnames, fig_pref

