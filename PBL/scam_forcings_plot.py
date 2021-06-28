import matplotlib.pyplot as mp
import numpy as np
import metpy.calc as mpy
import datetime as dt

## Quick plots ##

def plot_slhflx(time,sph,lhflx,shflx):
    mp.plot((time/sph)-6,lhflx[:,0,0]) 
    mp.plot((time/sph)-6,shflx[:,0,0]) 


    mp.xlabel('Local Time (hour)') ; mp.ylabel('W/m$^2$') 
    mp.title('Sensible/Latent Heat Flux') 


# Fig to file.
    mp.savefig(case_iop['iop_file_out']+'_lxflx_shflx.png', dpi=300)    


## Quick plots ##


def plot_profile ():

    mp.ion()

    fig1,axs = mp.subplots(nrows=1, ncols=5,figsize=(15, 5))

    mp.suptitle(iop_case+' Initial Conditions')

    axs[0].set_xlabel('K') ; axs[0].set_ylabel('m') ; axs[0].set_title('Temperature')
    axs[1].set_xlabel('K') ; axs[1].set_ylabel('') ; axs[1].set_title('Potential Temperature') ; axs[1].axes.yaxis.set_visible(False)
    axs[2].set_xlabel('g/kg') ; axs[2].set_ylabel('')  ; axs[2].set_title('Specific Humditiy'); axs[2].axes.yaxis.set_visible(False)
    axs[3].set_xlabel('%') ; axs[3].set_ylabel('') ; axs[3].set_title('Relative Humidity'); axs[3].axes.yaxis.set_visible(False)
    axs[4].set_xlabel('K') ; axs[4].set_ylabel('') ; axs[4].set_title('Liq Water Pot. Temperature'); axs[4].axes.yaxis.set_visible(False)


    yr0 = -100.
    yr1 = 3000.

    axs[0].set_ylim([yr0,yr1]) ; axs[1].set_ylim([yr0,yr1]) ; axs[2].set_ylim([yr0,yr1]) ; axs[3].set_ylim([yr0,yr1])
    axs[0].set_xlim([-5.,25.]) ; axs[1].set_xlim([295,310]) 

    axs[0].plot(tempc_plevs.transpose(), z_plevs) ;  axs[0].vlines(0, 0, z_plevs.max(), linestyle="dashed",lw=1) 
    axs[1].plot(the.transpose(), z_plevs) 
    axs[2].plot(q_plevs.transpose(), z_plevs)  ; axs[2].vlines(0, 0, z_plevs.max(), linestyle="dashed",lw=1)
    axs[3].plot(rh_plevs.transpose(), z_plevs) ; axs[3].vlines(0, 0, z_plevs.max(), linestyle="dashed",lw=1)  
    axs[4].plot(thel_plevs.transpose(), z_plevs)  
 
    mp.savefig(case_iop['iop_file_out']+'_IOP_ICs.png')
