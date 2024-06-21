


def sim_names_get(ctype):


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
	#cnames = ['CAM6 (L58)','KEparcel (L58)','KEparcel (L48)','KEparcel+PBLparcel (L48)']

	#cases = ['FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.001','FSCAM.T42_T42.arm97.cam_ke.001.L48']
	#cnames =  ['CAM6 (L58)','KEparcel (L58)','KEparcel (L48)']

	#cases = ['FSCAM.T42_T42.togaII.cam_ke.001','FSCAM.T42_T42.togaII.cam_ke.001.L48','FSCAM.T42_T42.togaII.cam_ke.001.L32']
	#cnames =  ['KEparcel (L58)','KEparcel (L48)','KEparcel (L32)']


	

	cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L48','FSCAM.T42_T42.togaII.cam_ke.000.L32','FSCAM.T42_T42.togaII.001.L256']
	cnames =  ['L58','L48','L32','L256']



	#cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L48','FSCAM.T42_T42.togaII.cam_ke.000.L32']
	#cnames =  ['L58','L48','L32']


	#cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L48','FSCAM.T42_T42.togaII.cam_ke.000.L32','FSCAM.T42_T42.togaII.cam_ke.001','FSCAM.T42_T42.togaII.cam_ke.001.L48','FSCAM.T42_T42.togaII.cam_ke.001.L32']
	#cases = ['FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.000.L48','FSCAM.T42_T42.arm97.cam_ke.000.L32','FSCAM.T42_T42.arm97.cam_ke.001','FSCAM.T42_T42.arm97.cam_ke.001.L48','FSCAM.T42_T42.arm97.cam_ke.001.L32']


	#cnames =  ['L58','L48','L32','KEparcel (L58)','KEparcel (L48)','KEparcel (L32)']


	#cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.002','FSCAM.T42_T42.togaII.cam_ke.003','FSCAM.T42_T42.togaII.cam_ke.004']

	#cases = ['FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.002','FSCAM.T42_T42.arm97.cam_ke.003','FSCAM.T42_T42.arm97.cam_ke.004']
	#cnames =  ['L58','KEparcel (L58)','PBL Parcel (L58)','KE/PBL Parcel (L58)']


	#cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke_cam6dev.000']
	#cnames =  ['CAM6+ZMpbl','CAM6-dev']


	#cases = ['FSCAM.T42_T42.arm97.cam_ke.003','FSCAM.T42_T42.arm97.cam_ke_cam6dev.000']
	#cnames =  ['CAM6+ZMpbl','CAM6-dev']

	#cases = ['FSCAM.T42_T42.togaII.cam_ke_cam6dev.000','FSCAM.T42_T42.togaII.cam_ke.003','FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L32','cam5.togaII','cam4.togaII','cam3.togaII']
	#cases = ['FSCAM.T42_T42.arm95.cam_ke_cam6dev.000','FSCAM.T42_T42.arm95.cam_ke.003','FSCAM.T42_T42.arm95.cam_ke.000','FSCAM.T42_T42.arm95.cam_ke.000.L32','cam5.arm95','cam4.arm95','cam3.arm95']
	#cnames = ['CAM6-dev','CAM6-L58-PBLpar','CAM6-L58','CAM6','CAM5','CAM4','CAM3']


	#cases = ['FSCAM.T42_T42.togaII.cam_ke_cam6dev.000','FSCAM.T42_T42.togaII.cam_ke.003','FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L32']
	#cases = ['FSCAM.T42_T42.arm95.cam_ke_cam6dev.000','FSCAM.T42_T42.arm95.cam_ke.003','FSCAM.T42_T42.arm95.cam_ke.000','FSCAM.T42_T42.arm95.cam_ke.000.L32']
	#cases = ['FSCAM.T42_T42.arm97.cam_ke_cam6dev.000','FSCAM.T42_T42.arm97.cam_ke.003','FSCAM.T42_T42.arm97.cam_ke.000','FSCAM.T42_T42.arm97.cam_ke.000.L32']
	#cnames = ['CAM6-dev','CAM6-L58-PBLpar','CAM6-L58','CAM6']

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



	#cases = ['FSCAM.T42_T42.togaII.cam_ke.000','FSCAM.T42_T42.togaII.cam_ke.000.L32','cesm1_cam5','cesm1_cam4','scam_undilute']
	#cnames = ['CAM6-L58','CAM6','CAM5','CAM4','CAM3']

	if ctype == 'cases' : ret_values = cases
	if ctype == 'cnames' : ret_values = cnames

	return ret_values
