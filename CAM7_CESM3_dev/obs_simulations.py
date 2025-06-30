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


    rl['CE3.112'] =  ['b.e30_alpha03d.BLT1850.ne30_t232_wgx3.112']
    rl['CE3.104'] = ['b.e30_beta02.BLT1850.ne30_t232.104']
    rl['CE3.98'] = ['b.e23_alpha17f.BLT1850.ne30_t232.098']
    rl['CE3.93'] =  ['b.e23_alpha17f.BLT1850.ne30_t232.093']

    rl['CE3.147'] =  ['b.e30_alpha06b.B1850C_LTso.ne30_t232_wgx3.147']
    rl['CE3.150'] =  ['b.e30_alpha06b.B1850C_LTso.ne30_t232_wgx3.150']
    rl['CE3.153'] =  ['b.e30_alpha06b.B1850C_LTso.ne30_t232_wgx3.153']
    rl['CE3.154'] =  ['b.e30_alpha06b.B1850C_LTso.ne30_t232_wgx3.154']

    rl['CE3.155'] =  ['b.e30_alpha06e.B1850C_LTso.ne30_t232_wgx3.155']    
    rl['CE3.156'] =  ['b.e30_alpha06e.B1850C_LTso.ne30_t232_wgx3.156']    
    rl['CE3.160'] =  ['b.e30_alpha06e.B1850C_LTso.ne30_t232_wgx3.160']  
    rl['CE3.162'] =  ['b.e30_alpha06e.B1850C_LTso.ne30_t232_wgx3.162']
    rl['CE3.163'] =  ['b.e30_alpha06e.B1850C_LTso.ne30_t232_wgx3.163']    

    rl['CE3.164'] =  ['b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.164']    
    rl['CE3.165'] =  ['b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.165']    
    rl['CE3.170'] =  ['b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.170']  
    rl['CE3.171'] =  ['b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.171']


    rl['C7d.dcs.600'] =  ['f.e30_beta04.FLTHIST.ne30.dcs600.001']
    rl['C7d.dp1.0.05'] =  ['f.e30_beta04.FLTHIST.ne30.dp1_0.05.001']
    rl['C7d.hscale.0.75'] =  ['f.e30_beta04.FLTHIST.ne30.hscale_0.75.001']
    rl['C7d.ke.2.5e-6'] =  ['f.e30_beta04.FLTHIST.ne30.ke_2.5.001']
    rl['C7d.c0ocn.0.045'] =  ['f.e30_beta04.FLTHIST.ne30.c0_ocn_0.045.001']

    rl['bline_rough'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.baseline_roughphis_SGH_grlnd28k.001']
    rl['lwdscale'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.lwdownscale_roughphis_SGH_grlnd28k.001']
    rl['tlapse_98'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.Tlapse0.0098_roughphis_SGH_grlnd28k.001']
    rl['tlapse_4'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.Tlapse0.004_roughphis_SGH_grlnd28k.001']
    rl['rainsnow'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.rainsnow_roughphis_SGH_grlnd28k.001']
    rl['bforest'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.borealforest_roughphis_SGH_grlnd28k.001']


# Surface bulk flux sensitivity runs.
    
    rl['eC7'] = ['f.cam6_4_032.FLTHIST_ne30.cam7_dev.002'] # Early CAM7 control
    rl['lhf'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.lhf_roughphis_SGH_grlnd28k.001']
    rl['flux_all_T2m'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.flux_all_T2m_roughphis_SGH_grlnd28k.001']
    rl['flux_all_Tsave'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.flux_all_Tsave_roughphis_SGH_grlnd28k.001']
    rl['G&G'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.baseline156_qsat.001']
    rl['ssq_T2m'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.ssq_T2m_roughphis_SGH_grlnd28k.001']
    rl['UA_flux'] = ['f.e30_cam6_4_078.FHISTC_LTso.ne30.ocnflx2_roughphis_SGH_grlnd28k.001']
        
    
    
# Pass through for obs.

    rl['HadISST'] = ['HadISST']
    rl['TRMM'] = ['TRMM']
    rl['GPCP'] = ['GPCP']
    rl['ERAI'] = ['ERAI']
    rl['ERS'] = ['ERS']
    rl['MERRA'] = ['MERRA']
    rl['JRA25'] = ['JRA25']
    rl['LARYEA'] = ['LARYEA']
    rl['ERS'] = ['ERS']
    rl['WHOI'] = ['WHOI']
    
    
    
# Data frame
    rl_df = pd.DataFrame.from_dict(rl, orient='index',columns=['run name'])
#    display(rl_df)
    return rl_df
