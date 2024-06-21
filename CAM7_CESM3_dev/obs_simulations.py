#####################################
# CAM6 Revert Experiments + others   
# ####################################


def cam_revert_list():

    import pandas as pd

    rl = {} # Revert List
# Releases
    rl['C4']   =  ['f40.1979_amip.track1.1deg.001']
    rl['C5']   =  ['30L_cam5301_FAMIP.001']
    rl['C6']   =  ['f.e20.FHIST.f09_f09.cesm2_1.001']
    rl['CC4']  =  ['b40.20th.track1.1deg.012']
    rl['CE1']  =  ['b.e11.B20TRC5CNBDRD.f09_g16.001']
    rl['CE2']  =  ['b.e21.BHIST.f09_g17.CMIP6-historical.001']
    rl['CE2.1850']  =  ['b.e20.B1850.f09_g17.pi_control.all.297']
    rl['CE1.1850']  = ['b.e11.B1850C5CN.f09_g16.005']
    rl['CC4.1850'] = ['b40.1850.track1.1deg.006']

    # Reverts
    rl['rC5now']  =   ['f.e20.FHIST.f09_f09.cesm2_1_cam5.001']
    rl['rC5']     =   ['f.e20.FHIST.f09_f09.cesm2_1_true-cam5.001']
    rl['rC5t']    =   ['f.e20.FHIST.f09_f09.cesm2_1_true-cam5_param_topo.001']
    rl['rUWold']  =   ['f.e20.FHIST.f09_f09.cesm2_1_uw.001']
    rl['rGW']    =   ['f.e20.FHIST.f09_f09.cesm2_1_iogw.001']
    rl['rZMc']  =   ['f.e20.FHIST.f09_f09.cesm2_1_capeten.001']
    rl['rMG1']  =   ['f.e20.FHIST.f09_f09.cesm2_1_mg1.001']
    rl['rSB']  =   ['f.e20.FHIST.f09_f09.cesm2_1_sb.002']
    rl['rTMS']  =   ['f.e20.FHIST.f09_f09.cesm2_1_tms.001']
    rl['rCE2i']  =   ['f.e20.FHIST.f09_f09.cesm2_1_revert125.001']
    rl['rC5p']  =   ['f.e20.FHIST.f09_f09.cesm2_1_revertcam5param.001']
    rl['rC5pm']  =   ['f.e20.FHIST.f09_f09.cesm2_1_revertcam5param.002']
    rl['rZMp']  =   ['f.e20.FHIST.f09_f09.cesm2_1_cam5_zmconv.001']
    rl['rM3']  =   ['f.e20.FHIST.f09_f09.cesm2_1_mam3.001']
    rl['rUW']  =   ['f.e20.FHIST.f09_f09.cesm2_1_uw.002']
    rl['rUWp']  =   ['f.e20.FHIST.f09_f09.cesm2_1_uw.003']
    rl['rMG1ii']  =   ['f.e20.FHIST.f09_f09.cesm2_1_mg1.002']
    rl['rice']  =   ['f.e20.FHIST.f09_f09.cesm2_1_ice-micro.001']
    rl['rpfrac']  =   ['f.e20.FHIST.f09_f09.cesm2_1_precip_frac_method.001']
    rl['rpremit']  =   ['f.e20.FHIST.f09_f09.cesm2_1_cld_premit.001']
    rl['rC5psalt']  =   ['f.e20.FHIST.f09_f09.cesm2_1_revertc5seasalt.001']
    rl['rC5pdust']  =   ['f.e20.FHIST.f09_f09.cesm2_1_revertc5dust.001']
    rl['rL30']  =   ['f.e20.FHIST.f09_f09.cesm2_1_L30.001']
    
# SST configs    
    rl['AMdsst'] = ['f.e20.FHIST.f09_f09.cesm2_1_reynolds_daily_sst.006']
    rl['CE2dsst']  =   ['f.e20.FHIST.f09_f09.cesm2_1_coupled-sst-amip_daily.001']
    rl['CE2sst']  =   ['f.e20.FHIST.f09_f09.cesm2_1_coupled-sst-amip.001']
    rl['CE2csst']  =   ['f.e20.FHIST.f09_f09.cesm2_1_coupled-sst-climo.001']
    
# Vertical resolution plots

# vres runs
    
    rl['C5']   =  ['30L_cam5301_FAMIP.001']
    rl['C6']   =  ['f.e20.FHIST.f09_f09.cesm2_1.001']
#    rl['L48']       =  ['f.e21.FWscHIST_BGC.ne30_ne30_mg17_L48_revert-J.001']
    rl['L48cin5']   =  ['f.e21.FWscHIST_BGC.ne30_ne30_mg17_L48_revert-J_num_cin-5.001']
    rl['L110'] = ['f.e21.FWscHIST_BCG.f09_f09_mg17_110L.001']
    rl['L48.BL10']=['f.e21.FWscHIST.ne30_L48_BL10_v3_tag20201111.001']
    rl['L48.BL10.zm1']=['f.e21.FWscHIST.ne30_L48_BL10_v3_tag20201111_zm1.001']
    rl['L48.BL10.zm2']=['f.e21.FWscHIST.ne30_L48_BL10_v3_tag20201111_zm2.001']
    
    
    rl['L32'] = ['f.e21.FWscHIST.ne30_L32_cam6_3_019_plus_CESM2.2.001.hf']
    rl['L48'] = ['f.e21.FWscHIST.ne30_L48_cam6_3_019_plus_CESM2.2.001.hf']
    rl['L58'] = ['f.e21.FWscHIST.ne30_L48_BL10_cam6_3_019_plus_CESM2.2.001.hf']
    rl['L58zm2'] = ['f.e21.FWscHIST.ne30_L48_BL10_cam6_3_019_plus_CESM2.2.001_zm2.hf']
    
    rl['L58bline'] = ['f.e21.FWscHIST.ne30_L48_BL10_cam6_3_041_control.hf.001']
    
   
  
    rl['L58zm2.numcin5'] = ["f.e21.FWscHIST.ne30_L48_BL10_cam6_3_019_plus_CESM2.2.001_zm2_numcin5.hf"]
    rl['cL58zm2'] = ['b.e21.BWsc1850.ne30_L48_BL10_cesm2_3_alpha05c_cam6_3_028_cam6_parcel_zm.004_zm2.hf']
    
    
    rl['L58zm2new'] = ['f.c6_3_41.FWscHIST.ne30_L58.zm2_fix.001']

#### CESM3 development

    rl['CE3.54'] = ['b.e23_alpha16g.BLT1850.ne30_t232.054']    # Cold branch
    rl['CE3.78'] = ['b.e23_alpha16g.BLT1850.ne30_t232.078']    # Cold branch
    rl['CE3.78b'] = ['b.e23_alpha16g.BLT1850.ne30_t232.078b']    # Cold branch + start from 75 at year 101
    
    rl['CE3.79'] = ['b.e23_alpha16g.BLT1850.ne30_t232.079']    # Cold branch + roughness
    rl['CE3.80'] = ['b.e23_alpha16g.BLT1850.ne30_t232.080']    # Warm branch
    rl['CE3.81b'] = ['b.e23_alpha16g.BLT1850.ne30_t232.081b']  # Cold branch + x3 new gustiness
    rl['CE3.82b'] = ['b.e23_alpha16g.BLT1850.ne30_t232.082b']  # Cold branch + x1 new gustiness
    rl['CE3.83b'] = ['b.e23_alpha16g.BLT1850.ne30_t232.083b']  # Cold branch + x1 new gustiness
    rl['CE3.90b'] = ['b.e23_alpha16g.BLT1850.ne30_t232.090b']  # 82b + 0.25*zm_pblh
    



    
# Pass through for obs.

    rl['TRMM'] = ['TRMM']
    rl['GPCP'] = ['GPCP']
    rl['ERAI'] = ['ERAI']
    rl['ERS'] = ['ERS']
    rl['MERRA'] = ['MERRA']
    
    
    
# Data frame
    rl_df = pd.DataFrame.from_dict(rl, orient='index',columns=['run name'])
    return rl_df
