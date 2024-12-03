'''
    ***********************************************************
    ***********************************************************
        PLOTTING ROUTINES
    ***********************************************************
    ***********************************************************
'''
import sys

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as mp
import matplotlib.colors as colors
import cartopy.crs as ccrs
from geocat.comp import interp_hybrid_to_pressure

from scipy.ndimage.filters import gaussian_filter

import vproc_setup as mysetup

dir_proot = '//glade/u/home/rneale/python/python-figs/vert_proc/'
fig_dpi = 150

''' 
    #########################################################
       SET OF REFERENCE P LEVELS
    #########################################################
'''

def clevs_ref():

    clevs = [1000, 975, 950, 925, 900, 850,  800, 750, 700, 
        600, 500, 450, 400,300, 250, 225,200, 175, 150, 125, 100,50]
    return clevs






'''
    #########################################################
        PLOT DIV AND OMEGA MAX/MIN LEVELS
    #########################################################
'''

def plot_div_pres(case_type,case,var_plt,varp_lev,da_in_ps,fls_ptr):


    
    '''
        Input Data Info
    '''
   
    print('-- Plotting pressure of minimum/maximum for ',var_plt)
    

    cc_pc = ccrs.PlateCarree(central_longitude=180)
    tcc_pc = ccrs.PlateCarree()


    # Plot layout
    nrows = 3
    ncols = 2
    
#	mp.figure(1)
    fig,axl =  mp.subplots(ncols=ncols,nrows=nrows,
                        subplot_kw={'projection': cc_pc},
                        figsize=(38,20))


    fig.patch.set_facecolor('white') 
    
    

    plevel = '500'
    season = 'DJF'



    axl=axl.flatten()
    
    
    
# Loop climo,lat,lon
    

    ''' SET UP PLOTTING STUFF '''


    

    
        
# Specific Plotting parmas.


    clevsp = [1008,992,962,938,912,875,825,775,725,650,550,475,425,350,275,232.5,212.5,187,162,132.5,112.5,75,25]

    clevsr = clevs_ref()
    clevsr.reverse()


    ccols =  ['lightgray','darkgray','gray','tan','khaki','yellow','gold','darkorange','lightsalmon','red','greenyellow',
              'green','darkgreen','lightseagreen','cyan','deepskyblue','blue','navy','purple','slateblue','violet','pink']
    ccols.reverse()
    cmap = colors.ListedColormap(ccols)

        

    mnames = ['Maximum','Minimum']	
    ens_ave	= ['Climatology','El Nino','La Nina']
    
        
    
    
    '''
        Loop Climo/Nino/Nina, may need to construct the pressure field if CAM.
    '''
    
    
    
    
    for iens,da_in in enumerate(varp_lev):
    
        
        if case_type != 'reanal':                      
            da_in = cam_lev2plev(da_in,da_in_ps[iens],fls_ptr)			
    
    
    
    # Find divergence from OMEGA and plot
    
#        if var_plt == 'DIV' and case_type != 'reanal':
#            print('DIV-plot')
#            da_in = -da_in.differentiate("lev",edge_order=2)
        
    
        for imm,mname in enumerate(mnames):
                
            print('  > '+mname+', '+ens_ave[iens])
 
 
# Get lev index of max/min value in a column + the array value at that level.      
            if mname == 'Maximum':
                da_plot = da_in.idxmax(dim='lev') 	
                da_lev_val = da_in.max(dim='lev')
                da_plot = da_plot.where(da_lev_val > 1.5e-4)

            if mname == 'Minimum':
                da_plot = da_in.idxmin(dim='lev')
                da_lev_val = da_in.min(dim='lev')
                da_plot = da_plot.where(da_lev_val < -1.5e-4)

# Normalize range for shading. Has to scale by std dev as climo. can be dominated by high values over topo.
            
#            lev_val_abs = np.fabs(da_lev_val)
#            max_val_abs = 0.5*max_val_abs/np.std(da_max_val)
#            print(np.std(da_max_val).values)
#            lev_val_norm= (lev_val_abs-np.min(lev_val_abs))/(np.max(lev_val_abs)-np.min(lev_val_abs))
#            lev_val_norm=np.clip(lev_val_norm,0,1)
            
            iplot = 2*iens+imm
                    
# Max/min level plotting
    
            axl[iplot].coastlines(color='black',linewidth=3)
    
            im = da_plot.plot.pcolormesh(ax=axl[iplot], transform=tcc_pc,levels=clevsp,cmap=cmap,rasterized=True,add_colorbar=False, shading='auto')
            
# Set opacity according to divergence magnitude at the max level.
                       
#            im.set_alpha(lev_val_norm)
            
            axl[iplot].set_title(ens_ave[iens]+' '+mname, fontsize=25)
            axl[iplot].hlines(0., -180, 180., color='black',lw=1,linestyle='--')

            


# Options for all plots.
    
    
    mp.subplots_adjust(bottom=0.25)

    fig.suptitle(case+' - Level of Maxium/Minimum',fontsize=50)
    clevst = [25.+ cc for cc in clevsr]

    cbar_ax = fig.add_axes([0.5, 0.34, 0.01, 0.46])

    cbar_ax.set_title('Max/Min Div. \n Pressure (mb)',fontsize=20)
    mp.colorbar(im, cax=cbar_ax, orientation="vertical",ticks=clevsr)
    cbar_ax.set_yticklabels(clevsr,fontsize=20)
    cbar_ax.invert_yaxis()

    mp.savefig(dir_proot+case+'_'+var_plt+'_minmax_level.png',dpi=fig_dpi)

    mp.show()


    

    
    
    
    
    
    
    
    
    '''
    #########################################################
        SCATTER PLOT OF TWO 2D FIELDS
    #########################################################
    '''
    
def scat_plot(case_type,case,var_cam,var2_cam,da_in_all,da2_in_all,da_in_ps,reg_df,fls_ptr):

    import seaborn as sb
    
    axs = mp.figure(figsize=(12,6))
    
    colors = ['r','b','g']
    cmaps  = ['blues','reds','oranges']
    
    var_df = pd.DataFrame()
    
    # Lev coordinate change

    tav_names = ['Seasonal','Nino34','Nina34']

    
    var_info = mysetup.vprof_set_vars()
     
    var1_lname = var_info.loc[var_cam]['long_name']   
    var2_lname = var_info.loc[var2_cam]['long_name']   
    
    var1_scale = var_info.loc[var_cam]['vscale']
    var2_scale = var_info.loc[var2_cam]['vscale']
        
    var1_units = var_info.loc[var_cam]['vunits']
    var2_units = var_info.loc[var2_cam]['vunits']
    
# Some names units scaling

    var_df = mysetup.vprof_set_vars()
    
    for itav, da_tav in enumerate(da_in_all): 
    
        da_in = da_in_all[itav]
        da2_in = da2_in_all[itav] 
    
        if case_type != 'reanal':   
            da_in = cam_lev2plev(da_in,da_in_ps[0],fls_ptr)	
            da2_in = cam_lev2plev(da2_in,da_in_ps[0],fls_ptr)	
            
        for ireg,reg in enumerate(reg_df.index):  ## 4 regions let's assume ##
    
            reg_name = reg_df.loc[reg]['long_name'] 
        
            reg_s = reg_df.loc[reg]['lat_s'] ; reg_n = reg_df.loc[reg]['lat_n']
            reg_w = reg_df.loc[reg]['lon_w'] ; reg_e = reg_df.loc[reg]['lon_e']

        
            print('  > Construct scatter plot for -- ',reg_name)

            da_reg = da_in.loc[:,reg_s:reg_n,reg_w:reg_e]
            da2_reg = da2_in.loc[:,reg_s:reg_n,reg_w:reg_e]
            
            var_x = da_reg.max(dim='lev').values.ravel()
            var_y = da2_reg.max(dim='lev').values.ravel()

            if var_cam[0] == 'DIV':
                var_x = da_reg.differentiate('lev').max(dim='lev').values.ravel()
            if var2_cam[0] == 'DIV':
                var_y = da_reg.differentiate('lev').max(dim='lev').values.ravel()	
            
            var_x = var_x*var1_scale
            var_y = var_y*var2_scale
                
            nlatlon = var_x.size
            
            var_df_reg = pd.DataFrame({'xvar':var_x[ip],'yvar':var_y[ip],'Region':reg_name} for ip in range(nlatlon))
            var_df = pd.concat([var_df,var_df_reg],ignore_index=True)
    
    
        print('  -- Plotting')
        
        yrange = [-0.03,0.13]
        xrange = [-1e-4,8e-4]
    
#    	yrange = [-0.12,0.04]
#		xrange = [-8e-4,1e-4]

        
        slevels = [0.02,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
#        sclip = ((yrange[0],yrange[1]),(xrange[0],xrange[1])) 
    
        mp.figure(figsize=(15,8))
        
#        axs = sb.kdeplot(var_df,x='xvar',y='yvar',hue='Region',levels=slevels,clip=sclip,common_norm=True)
        axs = sb.kdeplot(var_df,x='xvar',y='yvar',hue='Region',levels=slevels,common_norm=True)

#	axs = sb.jointplot(var_df, kind="kde",x='xvar',y='yvar',hue='Region',levels=slevels,clip=sclip,common_norm=True)
    
        axs.axhline(0.0,color='k',linestyle='--')
        axs.axvline(0.0,color='k',linestyle='--')
        
        mp.setp(axs.get_legend().get_texts(), fontsize='20') # for legend text
        mp.setp(axs.get_legend().get_title(), fontsize='28') # for legend title

        

        
#		sb.move_legend(axs, "lower right")
    
#		mp.hlines(0., xrange[0],xrange[1], color='black',lw=1,linestyle='--')
#		mp.vlines(0., yrange[0],yrange[1], color='black',lw=1,linestyle='--')
    
        mp.xlabel('Maximum '+var1_lname+' ('+var1_units+')',fontsize=20)
        mp.ticklabel_format(axis='y', style='sci', scilimits=(1,4))
        mp.xlim(xrange)
    
        mp.ylabel('Maximum '+var2_lname+' ('+var2_units+')',fontsize=20)
        mp.ylim(yrange)
    
        mp.suptitle(case+' - '+tav_names[itav],fontsize=20)
        mp.savefig(dir_proot+case+'_'+tav_names[itav]+'_min_scatter.png', dpi=fig_dpi)
    
        mp.show()


            
            
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    #########################################################
        CONVERT CAM LEV TO PLEV COORDINATE
    #########################################################
    '''
    
def cam_lev2plev(da_in,da_in_ps,fls_ptr):
    
    from geocat.comp import interp_hybrid_to_pressure
    
    
    
#### Change to Pressure vertical coordinate	
    
    
    p0 = 100000  # surface reference pressure in Pascals

# Specify output pressure levels
    new_levels = np.array(clevs_ref())
    new_levels = new_levels * 100  # convert to Pascals


# Extract the data needed
    
    hyam = fls_ptr['hyam']  # hybrid A coefficient
    hybm = fls_ptr['hybm']  # hybrid B coefficient


    if hyam.ndim == 2: hyam = hyam[0]
    if hybm.ndim == 2: hybm = hybm[0]


# Interpolate pressure coordinates form hybrid sigma coord
    

    da_in = interp_hybrid_to_pressure(da_in,
                      da_in_ps,
                      hyam,
                      hybm,
                      p0=p0,
                      new_levels=new_levels,
                      method='log')
# Swap variable name
    da_in = da_in.rename({'plev': 'lev'})

# Rescale to mb
    da_in = da_in.assign_coords(lev=0.01*da_in.lev)


    return da_in







            
    
    
    '''
    #########################################################
        CUSTONIZE LEGEND FOR VERTICAL PROFILE LINE PLOTS
    #########################################################
    '''
    
def leg_vprof(cases,case_type):

    from matplotlib.lines import Line2D
###		

#	all_colors = ['blue','red','green','purple','cyan','brown','yellow','orange','pink']
    all_colors = ['blue','orange','green','red','purple','cyan','brown','yellow','orange','pink']
    
    all_lstyles = ['-','--','-.',':']

    mod_ens = ['lens1','lens2','lense2','c6_amip']


    leg_elements = []
    leg_labels = []
    
    pmark,lcolor,lwidth,lstyle = [],[],[],[]

    icc,ibline = 0,0

    print(' -- Constructing Custom Legend for Vertical Profile Line Plots --')
    
    
    # LOOP CASES #
    
    print(cases)
    print(case_type)
    
    
    for ic,case in enumerate(cases):
        
        
        if case_type[ic] == 'reanal':
            pm,lc,lw,ls  = ('x',all_colors[icc],3,'-')
#			pm,lc,lw,ls  = ('x','Black',3,all_lstyles[ibline])
            icc+=1
            ibline+=1

        if case_type[ic] == 'lens1':
            pm,lc,lw,ls  = (None,'red',1,'-')  

        if case_type[ic] == 'lens2': 
            pm,lc,lw,ls  = (None,'blue',1,'-')  

        if case_type[ic] == 'lense2':
            pm,lc,lw,ls  = (None,'green',1,'-')  

        if case_type[ic] == 'lens2': 
            pm,lc,lw,ls  = (None,'blue',1,'-')          
            
        if case_type[ic] == 'c6_amip': 
            pm,lc,lw,ls  = (None,'blue',1,'-')  

        if case_type[ic] == 'lense2': 
            pm,lc,lw,ls  = (None,'green',1,'-')  

        if case_type[ic] in ['cam6_revert','cesm3_dev']:
            pm,lc,lw,ls  = ('.',all_colors[icc],1,'-')		
            icc+=1

        pmark.append(pm) ; lcolor.append(lc) ; lwidth.append(lw) ; lstyle.append(ls)

        
    # Only add first accurrence to the legend (mostly to lens members)
    
        
        if case_type.tolist().index(case_type[ic]) == ic or case_type[ic] not in mod_ens:
            
            leg_elements.append(Line2D([0], [0], marker=pmark[ic],color=lcolor[ic], lw=lwidth[ic], ls=lstyle[ic]))
            
            if case_type[ic] in mod_ens:
                leg_labels.append(case_type[ic])
            else:
                leg_labels.append(case)
                
        
    
    return leg_elements,leg_labels,pmark,lcolor,lwidth,lstyle
            

    

'''
    ###################################################################
            PLOT REGIONS WHERE VERTICAL PROFILES ARE TAKEN FROM
    ###################################################################
    '''
    
def vprof_reg_plot(reg_df):
    
    
    import matplotlib.patches as mpatches

    import cartopy.crs as ccrs
    from cartopy.feature import LAND

    desired_proj = ccrs.PlateCarree(central_longitude=180.)

    fig = mp.figure(figsize=(10,10))
    ax = mp.subplot(projection=desired_proj)

    ax.set_global()
    ax.set_extent([80, 280, -20, 50])
    facecolors = ['b','darkorange','g']
    
    reg_all = reg_df.index 
#	reg_name = reg_df.loc[reg]['long_name'] 	

    print('+++++++++++  REGIONS  ++++++++++++')
    print(*reg_all.values)
    

    
    for ireg,reg in enumerate(reg_all):
    

        reg_s = reg_df.loc[reg]['lat_s'] ; reg_n = reg_df.loc[reg]['lat_n']
        reg_w = reg_df.loc[reg]['lon_w'] ; reg_e = reg_df.loc[reg]['lon_e']
    
        dreg_lat = reg_n-reg_s 
        dreg_lon = reg_e-reg_w
    
        ax.add_patch(mpatches.Rectangle(xy=[reg_w, reg_s], width=dreg_lon, height=dreg_lat,
                                    facecolor=facecolors[ireg],
                                    alpha=0.2,
                                    transform=ccrs.PlateCarree()))
        
        
    ax.gridlines()
    ax.coastlines()
#	ax.set_xlim([80,300])
    ax.add_feature(LAND,color='k')

    mp.savefig(dir_proot+'test_region.png', dpi=fig_dpi, bbox_inches='tight') 
    mp.show()
    
        
    
    
    
    
    
'''
#########################################################
    PLOTTING VERTICAL PROFILES SET
#########################################################
'''
    
def vprof_clim_nino(vproc_cases,p_levs,var_cam,reg_df,var_df,case_type,case_desc,yrs,pref_out):
    

    nino_states = ['Climo ('+str(yrs[0])+'-'+str(yrs[1])+')','Nino Anomalies','Nina Anomalies']
    nino_colors = ['k','r','b']
    nnino = len(nino_states)
    
    
    ''' Plot line for this case and for this region '''
    
    case_names = list(vproc_cases.keys())
    reg_names = reg_df.index
       
    ncases = len(case_names)
    nreg = len(reg_names)
    

    ''' Figure Out Legend and Line Colors based '''

    print('-Plotting cases:',case_names)
    print('-Plotting regions:',reg_names)

    ''' Legend and line resources '''
    
    leg_elements,leg_labels,pmark,lcolor,lwidth,lstyle = leg_vprof(case_desc,case_type)
    lloc = 'lower right' if var_cam in ['ZMDQ','STEND_CLUBB'] else 'lower left' 

    
    ''' Specifics for this variable '''
    xmin = var_df.loc[var_cam]['xmin'] ; xmax=var_df.loc[var_cam]['xmax']
    axmin = var_df.loc[var_cam]['axmin'] ; axmax=var_df.loc[var_cam]['axmax']                     
    vunits = var_df.loc[var_cam]['vunits'] 
    var_text = var_df.loc[var_cam]['long_name']   

    print('')
                    
    fign, axn = mp.subplots(nreg,3,figsize=(26, 26))  

    lrange_plot = False

    for icase,case in enumerate(case_names):

        print('V. profile plot for case = ',case)
    
        for ireg,region in enumerate(reg_names):

            reg = reg_df.index[ireg] 

            reg_name = reg_df.loc[reg]['long_name'] 
       
            reg_s = reg_df.loc[reg]['lat_s'] ; reg_n = reg_df.loc[reg]['lat_n']
            reg_w = reg_df.loc[reg]['lon_w'] ; reg_e = reg_df.loc[reg]['lon_e']
    
#            print()
#            print('-- Region = ',reg_name,' - ',reg_s,reg_n,reg_w,reg_e)
    
            reg_a_str = '%d-%d\u00b0E %.1f-%d\u00b0N' % (reg_w,reg_e,reg_s,reg_n)
            reg_a_out = '%d-%dE_%.1f-%dN' % (reg_w,reg_e,reg_s,reg_n)  
    
        
            for inino,nino_reg in enumerate(nino_states):

                
                # Entry is a little tricky 3 entries (climo,nino,nina) per successive region  
                idata_pos = ireg*nnino+inino
                var_fig = vproc_cases[case][idata_pos]
                var_fig_p = var_fig.lev


# Just plot the average and shading between min and max for each reg/nino on the count for the last case.

                if lrange_plot and icase == ncases-1:
                    # Order all case profiles for this reg/nino
                    profs_ens = []
#                    print( vproc_cases.items())
                    for key, this_prof in vproc_cases.items() :
                        profs_ens.append(this_prof[idata_pos])
                    profs_ens_arr = xr.concat(profs_ens, dim='case')
                 

                    vprof_mean = profs_ens_arr.mean(dim='case')
                    vprof_range_min =  profs_ens_arr.min(dim='case')
                    vprof_range_max =  profs_ens_arr.max(dim='case')
                    
                    print('-Ensemble plotting',icase,ireg,inino)
                    axn[ireg,inino].plot(vprof_mean,var_fig_p,lw=5,color=lcolor[0],fillstyle='none') 
#                    axn[ireg,inino].plot(vprof_range_min,var_fig_p,lw=5,color='red',fillstyle='none') 
#                    axn[ireg,inino].plot(vprof_range_max,var_fig_p,lw=5,color='red',fillstyle='none') 
                    axn[ireg,inino].fill_betweenx(var_fig_p, vprof_range_min,vprof_range_max,color=lcolor[0],alpha=0.35)
                    print('-Done...')
                
                if not lrange_plot:    
                    axn[ireg,inino].plot(var_fig,var_fig_p,lw=lwidth[icase],markersize=9,marker=pmark[icase],color=lcolor[icase],linestyle=lstyle[icase])  
                    
                del(var_fig)

                
                if (icase==0) :
                    axn[ireg,inino].set_title(nino_states[inino],fontsize=20,color=nino_colors[inino])
                    axn[ireg,inino].set_ylabel('mb',fontsize=16) 
                    axn[ireg,inino].set_xlabel(vunits,fontsize=16)      
                    axn[ireg,inino].set_yticks(p_levs)
                    axn[ireg,inino].set_yticklabels(p_levs,fontsize=14)
                    axn[ireg,inino].invert_yaxis()
            
#                    axn[ireg,0].set_xticklabels(np.arange(xmin,xmax,0.1*(xmax-xmin)),fontsize=12)
            
                    axn[ireg,inino].tick_params(axis='both', which='major', labelsize=14)
                    axn[ireg,inino].grid(linestyle='--')  

                    axn[ireg,0].set_xlim([xmin,xmax])
                    axn[ireg,1].set_xlim([axmin,axmax])
                    axn[ireg,2].set_xlim([axmin,axmax])

                    
                    axn[ireg,0].legend(leg_elements,leg_labels,fontsize=15,loc = lloc)
                    axn[ireg,0].legend(leg_elements,leg_labels,fontsize=15,loc = lloc)
                    
                    axn[ireg,0].text(0, 1, reg_name, transform=axn[ireg,0].transAxes, ha='left', va='top', fontsize=20)
                    axn[ireg,0].text(0, 0.95, reg_a_str, transform=axn[ireg,0].transAxes, ha='left', va='top', fontsize=16)
                    
                   
                    
#                    if ((var_fig.values.min < 0) and (var_fig.max > 0)) :
                    axn[ireg,inino].vlines(0., max(p_levs), min(p_levs), linestyle="--",lw=1, color='black')

  
     # Main title
    fign.suptitle('ENSO Anomalies - '+var_text,fontsize=20)
    
    mp.rcParams['xtick.labelsize'] = 15 # Global set of xtick label size    
    
    
    
    
    # Hard copy  
    mp.show()
    fign.savefig(dir_proot+pref_out+'_nino_vprof_'+var_cam+'_'+str(yrs[0])+'_to_'+str(yrs[1])+'.png', dpi=80)

    