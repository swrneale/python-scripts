import numpy as np

'''
    BAR PLOT OF 1D TIME AVERAGES
'''



def scam_bar(ax,pvar1d_all,cnames_all,units1d):
    
    pvar_bar = np.nanmean(pvar1d_all, axis=1)
    ax.bar(cnames_all,pvar_bar)
    ax.set_ylabel(units1d,fontsize=10) 
    ax.set_xticklabels(cnames_all,rotation=90.)

    return
