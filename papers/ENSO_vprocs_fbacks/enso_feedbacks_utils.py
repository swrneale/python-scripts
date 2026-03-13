'''
    Utility Routines for ENSO Diagnostics (currently a redo of the ncl quick look panels)
'''

import xarray as xr
import numpy as np
import pandas as pd

import geocat.comp as gc

import glob as glob
import os as os



### Mya need to CHANGE for your own local directory to write out derived timeseries (not used for LENS, already in tseries format.)
dir_ncout = '/glade/work/rneale/python-netcdf/enso/'
dir_data = '/glade/work/rneale/data/'



''' Set nino123456 domain '''

def nino_region(nino_name):

# Same lat range for now.cesm1
    nino_s = -5. ;  nino_n = 5.

    match (nino_name):
        case 'nino3':  nino_w = 210 ;  nino_e = 270.
        case 'nino34': nino_w = 190 ;  nino_e = 240.
        case 'nino4':  nino_w = 160 ;  nino_e = 210.
            


    return nino_w,nino_e,nino_n,nino_s





























''' Check Requested Time Overlap and Trim to a Common Period '''


def dataset_ts_trim(da_x,da_y,yr0,yr1,var_x,var_y,get_months):


# Time details for x axis

    print('  - Requested variable year range = ',yr0,'-',yr1)
    
    print('  - Time details for x-axis - '+var_x)
    fyr0 = da_x.time.min().dt.strftime('%Y.%b').item()
    fyr1 = da_x.time.max().dt.strftime('%Y.%b').item()
    print('    - Available year range = ',fyr0,'-',fyr1)

    print('  - Time details for y-axis - '+var_y)
    fyr0 = da_y.time.min().dt.strftime('%Y.%b').item()
    fyr1 = da_y.time.max().dt.strftime('%Y.%b').item()

    da_x_min = da_x.time.min().values
    da_x_max = da_x.time.min().values
    da_y_min = da_y.time.min().values
    da_y_max = da_y.time.min().values
    
    print('    - Available year range = ',fyr0,'-',fyr1)
    

    if (da_x_min != da_x_max and da_y_min != da_y_max):
        print('    - DataArray time range of axes do not match')


# Subset time range (years, seasons) from available full range. Used padded 4 digit integers.


    tx_mask = ((da_x.time.dt.year >= yr0) & (da_x.time.dt.year <= yr1) &
                    (da_x.time.dt.month.isin(get_months)))
    
    da_x = da_x.sel(time=tx_mask)

    
    ty_mask = ((da_y.time.dt.year >= yr0) & (da_y.time.dt.year <= yr1) &
                    (da_y.time.dt.month.isin(get_months)))

    da_y = da_y.sel(time=ty_mask)


# Check they have the same time dimenstion noe.

    


#    print(da_y.time.values)
#    x0, x1 = pd.to_datetime(da_x.time.values[0]),pd.to_datetime(da_x.time.values[-1])
#    y0, y1 = pd.to_datetime(da_y.time.values[0]),pd.to_datetime(da_y.time.values[-1])

#    span_x = x1.year - x0.year + 1
#    span_y = y1.year - y0.year + 1

 #   print('     - Subsetted data ime span is ',[span_y == span_x])
    
       
    return da_x, da_y

















def bin_mean_by_level(
    field_ts,        # (time, lev)
    index_ts,        # (time,)
    bins,
    bin_centers,
    lev_dim="plev",
    time_dim="time",
):
    """
    Returns DataArray with dims (lev, bin)
    """

    out = []

    for lev in field_ts[lev_dim].values:
        y = field_ts.sel({lev_dim: lev}).values
        x = index_ts.values

        bin_means = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (x >= b0) & (x < b1)
            if mask.any():
                bin_means.append(np.nanmean(y[mask]))
            else:
                bin_means.append(np.nan)

        da = xr.DataArray(
            bin_means,
            coords={"bin": bin_centers},
            dims=["bin"],
        ).expand_dims({lev_dim: [lev]})

        out.append(da)

    return xr.concat(out, dim=lev_dim)







def bin_mean_var_by_level(
    field_ts,        # (time, lev)
    index_ts,        # (time,)
    bins,
    bin_centers,
    lev_dim="plev",
    time_dim="time",
):
    """
    Returns:
        mean_da : DataArray (lev, bin)
        var_da  : DataArray (lev, bin)
    """

    mean_out = []
    var_out  = []

    for lev in field_ts[lev_dim].values:
        y = field_ts.sel({lev_dim: lev}).values
        x = index_ts.values

        bin_means = []
        bin_vars  = []

        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (x >= b0) & (x < b1)

            if mask.any():
                vals = y[mask]
                bin_means.append(np.nanmean(vals))
                bin_vars.append(np.nanstd(vals, ddof=1))  # sample variance
            else:
                bin_means.append(np.nan)
                bin_vars.append(np.nan)

        mean_da = xr.DataArray(
            bin_means,
            coords={"bin": bin_centers},
            dims=["bin"],
        ).expand_dims({lev_dim: [lev]})

        var_da = xr.DataArray(
            bin_vars,
            coords={"bin": bin_centers},
            dims=["bin"],
        ).expand_dims({lev_dim: [lev]})

        mean_out.append(mean_da)
        var_out.append(var_da)

    mean_da = xr.concat(mean_out, dim=lev_dim)
    var_da  = xr.concat(var_out,  dim=lev_dim)

    return mean_da, var_da


















def nino_anom_ts(da_axis, nino_reg, axis_vals,lev_dim='plev'):

    
    """
    Calculate Niño anomaly time series and PDFs (supports 2D or 3D data).

    Parameters
    ----------
    da_axis : xarray.DataArray
        Monthly data (2D or 3D with lev dimension)
    nino_reg : str
        Niño region key (e.g., "nino34")
    axis_vals : 1D array
        x-axis values for PDF
    lev_dim : str, optional
        Name of vertical dimension (default: 'lev')

    Returns
    -------
    nino_ts  : xarray.DataArray
        Niño anomaly time series
        - shape (time) for 2D
        - shape (lev, time) for 3D
    nino_pdf : xarray.DataArray
        PDF(s)
        - shape (axis) for 2D
        - shape (lev, axis) for 3D
    """

    from scipy.stats import gaussian_kde

   
    # ---- Niño region bounds ----
    nino_w, nino_e, nino_n, nino_s = nino_region(nino_reg)

    # ---- Subset region ----
    nino_axis = da_axis.sel(
        lat=slice(nino_s, nino_n),
        lon=slice(nino_w, nino_e)
    )

    # ---- Area-weighted mean ----
    weights = np.cos(np.deg2rad(nino_axis.lat))
    nino_waxis = nino_axis.weighted(weights).mean(dim=["lat", "lon"])

    # ---- Monthly climatology & anomalies ----
    clim = nino_waxis.groupby("time.month").mean("time")
    nino_anom = nino_waxis.groupby("time.month") - clim


    # ---- TIME SERIES OUTPUT ----
    nino_ts = nino_anom

    # ---- PDF helper ----
    def _kde_da(x, lev_val=None):
        x = x[np.isfinite(x)]
        if np.all(x == 0): # Don't kde if all values are zero (e.g., CLOUD in the stratosphere)
            pdf_vals = np.full_like(axis_vals, np.nan)
        else :
            kde = gaussian_kde(x, bw_method=0.25)
            pdf_vals = kde(axis_vals)

        da = xr.DataArray(
            pdf_vals,
            coords={"axis": axis_vals},
            dims=["axis"]
        )

        if lev_val is not None:
            da = da.expand_dims({lev_dim: [lev_val]})

        return da

    # ---- PDF construction ----
    if lev_dim in nino_anom.dims:
       
        pdf_list = []
        for lev in nino_anom[lev_dim].values:
            x = nino_anom.sel({lev_dim: lev}).values.flatten()
            pdf_list.append(_kde_da(x, lev_val=lev))

        nino_pdf = xr.concat(pdf_list, dim=lev_dim)
    else:
        x = nino_anom.values.flatten()
        
        nino_pdf = _kde_da(x)

        
    return nino_ts, nino_pdf








''' Calculate El Nino Anomalies timeseries'''


def nino_anom_ts_2d(da_axis,nino_reg,axis_vals):

    from scipy.stats import gaussian_kde


# nino and var regions
# # Taking settings from the Clivar 2020 ENSO metrics


    nino_w,nino_e,nino_n,nino_s = nino_region(nino_reg)

    
    nino_axis = da_axis.sel(lat=slice(nino_s, nino_n), lon=slice(nino_w, nino_e))
    
    
    # Step 3: Area-weighted average over lat/lon
    weights = np.cos(np.deg2rad(nino_axis.lat))
    
    nino_waxis = nino_axis.weighted(weights).mean(dim=["lat", "lon"])
    
    
    # Group by month, calculate climatology (mean for each calendar month)
    nino_caxis = nino_waxis.groupby('time.month').mean('time')

    
    # Subtract monthly climatology to get anomalies
    
    nino_anom = nino_waxis.groupby('time.month') - nino_caxis

    
    # Convert to NumPy and flatten, mask NaNs
    
    nino_1d = nino_anom.values.flatten()

    nino_1d = nino_1d[~np.isnan(nino_1d)] 


    
    # Strip out requested season before constructing PDFs

    ''' PDFs of monthly variable '''

    nino_kde = gaussian_kde(nino_1d, bw_method=0.3)

    nino_pdf = nino_kde(axis_vals)

  
    

    return nino_1d, nino_pdf









'''
    PLOTTING FUNCTIONS
'''


def plot_latlon(ax,fig,icase,var,pvar,vunits,case,cname,seas,pregion, nino_reg,mask_ocn,lanom_plot,last_plot, fscale):

    from matplotlib import cm
    import matplotlib.pyplot as mp

    
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature # Map features
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter



    preg_area = get_pregion(pregion)

    lat_min = preg_area[pregion]['lat_min']
    lat_max = preg_area[pregion]['lat_max']
    lon_min = preg_area[pregion]['lon_min']
    lon_max = preg_area[pregion]['lon_max']
    

    pvar = pvar.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)) 
  
    
    dpproj = ccrs.PlateCarree()
    
    if pregion in ['NPolar','SPolar']: 
        pproj = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70.0) # For Northern Hemisphere
        

        # ---- Create a circular boundary for Polar plots ----
        theta = np.linspace(0, 2 * np.pi, 100)
        center = np.array([0.5, 0.5])  # center of the axes
        radius = 0.5
        circle = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle_path = mpath.Path(circle * radius + center)
    
        ax.set_boundary(circle_path, transform=ax.transAxes)

    

    
    else:

        pproj = ccrs.PlateCarree()
        pproj = ccrs.Robinson() if pregion == 'Global' else ccrs.PlateCarree()

        if pregion != 'Global':
        
            ax.set_yticks(np.arange(lat_min, lat_max + 10.0, 10.0), crs=pproj)
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)

            ax.set_xticks(np.arange(lon_min, lon_max + 30.0, 30.0), crs=pproj)
            lon_formatter = LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)

  
    # Plotting

 
#    norm = cm.colors.BoundaryNorm(boundaries=levs_4colbar, ncolors=256)


    # Plotting (first enable NaNs to be masked out in gray)
    # -- Not sure if this is working yet --
    cmap =  'RdBu_r'
    levels = np.array([-50.,-40.,-30, -25, -20, -15, -10,  0, 10, 15, 20, 25, 30,40,50.])*1.
   
#    cmapo = mp.get_cmap(cmap).copy()
#    cmapo.set_bad('lightgray')  # Plot light gray for NaNs

    cmapo = mp.get_cmap(cmap)   # or your cmap variable
    cmapo = cmapo.with_extremes(bad="lightgray")   # set color for NaNs

    pvarz = pvar
    pvarz = np.ma.masked_inside(pvar, -0.5, 0.5)

    pplot = ax.contourf(pvar.lon, pvar.lat, pvarz, transform=dpproj, levels=levels,extend='both', cmap=cmapo)


    # Add color bar before solid countours
        
#    if icase == 0 or (icase == 1 and lanom_plot) :
    cbar_area = [1.02, 0.05, 0.05, 0.85] ; cbar_orient = "vertical"
    cbar_area = [0.02, -0.15, 0.9, 0.05 ] ; cbar_orient = "horizontal"
    cbar = ax.inset_axes(cbar_area, transform=ax.transAxes)
    cbar = fig.colorbar(pplot, cax=cbar, orientation=cbar_orient)
    cbar.set_label(vunits)


    
  
    levels_nozero = [l for l in levels if l != 0]
    pplot = ax.contour(pvar.lon, pvar.lat, pvar, transform=dpproj, levels=levels_nozero,colors='black', linewidths=0.4)

    ax.set_title(r"$\bf{"+case+"}$ - "+var+"  ("+nino_reg+") - "+seas) 
    ax.text(
            0.01, 0.99, cname,   # y < 0 puts it below xlabel
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=10*fscale
        )

# Mapping
    
#    ax.set_extent([lon_min, lon_max, lat_min, lat_max], pproj)

    
    ax.coastlines(linewidth=1,color='black',resolution='110m')
    ax.gridlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0, linewidth=1.2)
#    ax.add_feature(cfeature.BORDERS.with_scale('110m'))
    ax.add_feature(cfeature.LAND, facecolor='silver')
#    coast = cfeature.NaturalEarthFeature('physical', 'coastline', '110m')
#    ax.add_feature(coast, linewidth=0.3)

    # Add U.S. states
    if pregion == "US":
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="black")

  



















''' Set some plot domain ranges for an axis '''

def fig_domains(vname):

    match vname:
        case 'TS':
            vmin,vmax = -4.,4.
        case 'PRECT':
            vmin,vmax = -4.,4.
        case 'TAUX':
            vmin,vmax = -0.08, 0.08
        case 'OMEGA500':
            vmin,vmax = -0.8, 0.8   
        case _ if vname in ['DTCOND500','DTCOND300','DTCOND700']:
            vmin,vmax = -3, 3   
           
        case _:
            
             print('  - Variable not recognized '+vname)

# Retun array where gaussian_kde will fit to.

    axis_vals = np.linspace(vmin, vmax, 100)

    return vmin,vmax,axis_vals








''' 
    Extract a single 2D pressure level slice forom the 3D variable (e.g., U200 fomr U) 
'''

def cam_2d_from_3d(var_2d,var_3dget,var_plev,files_suff=None,ds_h0=None):

# Variable requiring more than 1 timeseries file

    var_derived = ['DIV']

    
    # Grab/read either data from CESM1/2 in single timeseries files or CESM3 data in a dataset of h0a files.

    ''' CESM1/2 (tseries)'''

    if files_suff is not None: # CESM1/2
    
        files_ps = files_suff.replace(var_2d, 'PS')
        print(files_ps)
        ds_ps = xr.open_mfdataset(files_ps,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")
        
        
        # Swap in the 3D variable from the existing 2D variable in the files_suff.
    
        if var_2d not in var_derived: # General 3D -> 2D map
    
            files_var = files_suff.replace(var_2d, var_3dget) 
            ds_var = xr.open_mfdataset(files_var,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")
            
            
    
        else: # Special cases
    
            if var_plev == 'DIV':
    
                files_u = files_suff.replace(var_2d, 'U')
                files_v = files_suff.replace(var_2d, 'V')
                
                ds_u = xr.open_mfdataset(files_u,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")
                ds_v = xr.open_mfdataset(files_v,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")
            
                da_uplev = get_plev_cam(ds_u,ds_ps,var_plev,'U') 
                da_vplev = get_plev_cam(ds_v,ds_ps,var_plev,'V') 
        
      
    
    ''' CESM3 (h0a)'''   
    
    
    if ds_h0 is not None:  # Mostly CESM3 development simulations (muich easier since we already have the Dataset.
    
        
        if var_2d not in var_derived:  # General 3D -> 2D map
    
            da_out = get_plev_cam(ds_h0a,ds_h0a,var_plev,var_3dget)            
    
        else: # Special cases
    
            if var_plev == 'DIV':
    
                da_uplev = get_plev_cam(ds_h0a,ds_h0a,var_plev,'U') 
                da_vplev = get_plev_cam(ds_h0a,ds_h0a,var_plev,'V') 
        
    
    
    return da_out











''' Interpolate a 3D field from CAM to a single pressure level(h0concatonation or timesries) '''


def get_plev_cam(ds_var,ds_ps,var_plev,var_2interp):   
    
    import geocat.comp as gc

    print('-Interpolating ',var_2interp,' to ',var_plev,' mb')


    da_var = ds_var[var_2interp]
 
    hyam = ds_var['hyam']  # hybrid A coefficiet
    hybm = ds_var['hybm']  # hybrid B coefficient
    p0 =  ds_var['P0']  # Reference pressure

    da_ps = ds_ps['PS']
    
    if hyam.ndim == 2: hyam = hyam[0]
    if hybm.ndim == 2: hybm = hybm[0]


    # Fix lev to not be chunked.
    da_var = da_var.chunk({"lev": -1})
    
    
    # Interpolate pressure coordinates form hybrid sigma coord
   
    da_var = gc.interp_hybrid_to_pressure(da_var,
                          da_ps,
                          hyam,
                          hybm,
                          p0=p0,
                          new_levels=var_plev,
                          method='log')
    # Rename and swap variable name
#    da_var = da_var.rename(var_2d).squeeze()
   
#    da_var = da_var.rename({'lev': 'plev'})
    
    # Rescale to mb
    da_var = da_var.assign_coords(plev=0.01*da_var.plev)

#        print('- Unlazying...')
#        with ProgressBar():
#           da_out = da_var.compute()  # Turns into in-memory NumPy-backed array


#        print('- Writing out...')
#        da_out.to_netcdf(out_dir+out_file,mode="w")

     
    print('Done')

    
    
    return da_var




''' 
    Grab Plot Region Domain
'''

def get_pregion(pregion):

  
    
    nrow = None
    ncol = None

    reg_info = {}
    
    match pregion:

        case 'Global': # Whole domain
     
            reg_info[pregion] = {
                'lat_min' : -90 , 'lat_max' : 90,
                'lon_min' : 0. , 'lon_max' : 360.,
                'plev_scale' : 0.2,
                'aplev_scale' : 0.2
            }

    
        case 'LabSea': # Labrador Sea
     
            reg_info[pregion] = {
                'lat_min' : 35 , 'lat_max' : 70,
                'lon_min' : 280 , 'lon_max' : 340,
                'plev_scale' : 0.2,
                'aplev_scale' : 0.2
            }
                
        case 'IO': # Indian Ocean 
            
             reg_info[pregion] = {
                'lat_min' : -10 , 'lat_max' : 35,
                'lon_min' : 50 , 'lon_max' : 120,
                'plev_scale' : 1.,
                'aplev_scale' : 1.
             }
    
        case 'US': # USA #
            
             reg_info[pregion] = {
                'lat_min' : 25 , 'lat_max' : 55,
                'lon_min' : -120 , 'lon_max' : -70, 
                'plev_scale' : 0.25,
                'aplev_scale' : 0.25
             }
    
        case 'SAm': # South America 
            
             reg_info[pregion] = {
                'lat_min' : -40 , 'lat_max' : 15,
                'lon_min' : 250 , 'lon_max' : 330, 
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
    
    
        case 'Aus': # Australia 
            
            reg_info[pregion] = {
                'lat_min' : -20 , 'lat_max' : 10,
                'lon_min' : 120 , 'lon_max' : 150,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
    
        case 'TP': # Tropical Pacific
            
             reg_info[pregion] = {
                'lat_min' : -10 , 'lat_max' : 10,
                'lon_min' : 120., 'lon_max' : 290.,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
    
             nrow = 5 ; ncol = 2 # Rows and columns
    
        case 'WP': # West Pacific
            
             reg_info[pregion] = {
                'lat_min' : -20 , 'lat_max' : 40,
                'lon_min' : 110 , 'lon_max' : 270.,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
            
        case 'IndoPac': # West Pacific
            
             reg_info[pregion] = {
                'lat_min' : -40 , 'lat_max' : 40,
                'lon_min' : 40 , 'lon_max' : 200.,
                'plev_scale' : 0.5,
                'aplev_scale' : 0.5
            }
            
        case 'Tropics': # Tropics Wide
            
             reg_info[pregion] = {
                'lat_min' : -45 , 'lat_max' : 45,
                'lon_min' : 0. , 'lon_max' : 360.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
             }

        case 'EPac': # Tropics Wide
            
             reg_info[pregion] = {
                'lat_min' : -30 , 'lat_max' : 30,
                'lon_min' : 200. , 'lon_max' : 280.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
             }

        case 'NPac': # Tropics Wide
            
             reg_info[pregion] = {
                'lat_min' : -10 , 'lat_max' : 80,
                'lon_min' : 120. , 'lon_max' : 310.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
             }

        case 'Boreal': # Tropics Wide
            
             reg_info[pregion] = {
                'lat_min' : 40 , 'lat_max' : 75,
                'lon_min' : 190. , 'lon_max' : 280.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
            }

    
        case 'NPolar': # Northern Polar (stereographic)
            
             reg_info[pregion] = {
                'lat_min' : 60. , 'lat_max' : 90.,
                'lon_min' : -180. , 'lon_max' : 180.,
                'plev_scale' : 0.8,
                'aplev_scale' : 1.
            }
        
    
    return reg_info





'''
    ERA Levels
'''

def grab_era5_levs():

# ERA5 levels from extracted data.
    elevs = np.array([1000, 925, 850, 700, 600, 500,
            400, 300, 250, 200, 150, 100,
            70, 50, 30, 20])*100.

    return elevs




''' Read in Different Datasets For each Axis '''



def get_dataset(case,case_type,var_axis,yr0,yr1,lread_in_all_hist,lwrite_ts_file ,lread_ts_file):

 
    cvars =      ['TS','TAUX','PRECT',   'OMEGA500', 'DTCOND300',' DTCOND500','DTCOND700','OMEGA','RELHUM','Q',    'DIV',  'Z3', 'U', 'V','T','CLOUD','DTCOND','DCQ'] 
    cvar_scales = [1.,  1.,  86400.*1000.   ,36.      ,86400.,     86400.      ,86400.,     36.,     1.,  1000.,   1.e+6,   1.,   1.,  1., 1., 100., 86400.,86400.*1000]
    cvar_scale = cvar_scales[cvars.index(var_axis)]

# Obs. variable names.
    evars = ['sst','avg_iews','tp',   'w',      'd',      'd',      'd', 'w', 'r', 'q', 'd', 'z', 'u','v','t','cc','mmpdt','mmpdq']
    efvars = ['sst','taux','prect','omega500','div200','div300','div400', 'omega', 'rh', 'q','div','z','u','v','t','cloud','dtdt_param','dqdt_param']


    lvar_from3d = False ; var_plev = False ; var_3dget = False

# Figure out if 2D variable can come from 3D input
    lvar_from3d = var_axis[-3:].isdigit() if len(var_axis) > 3 else False # Test if last 3 digits are an interger and so likely a 2D variable that came from a 3D variable.
  
    if (lvar_from3d):
        var_plev = var_axis[-3:]        # Grab the pressure level if the above is true
        var_3dget = var_axis[:-3]
        
  

# Read in data according to case type and source
    
    match case:

    
        case _ if case_type == 'OBS':


            # Selct obs. source.

            print('  - Grabbing ',case,' data for',var_axis)
            
            match case:

                case 'ERA5':

                    
                    
                    dir_era5 = '/glade/derecho/scratch/rneale/ERA5/mmean/1deg/'


# Some ERA5 variable mappings to CESM vars.

                    evar = evars[cvars.index(var_axis)]
                    efvar = efvars[cvars.index(var_axis)]
                    
                    da_axis = xr.open_dataset(dir_era5+efvar+'/'+efvar+'_era5_monthly_1x1.nc')[evar]
                    
                    if 'valid_time' in da_axis.dims:
                        da_axis = da_axis.rename({'valid_time': 'time'})

                    # Change pressure level name.
                    da_axis = da_axis.rename({"pressure_level": "plev"}) if "pressure_level" in da_axis.dims else da_axis

                    vscale = 1.

                   
                    if var_axis in ['TS','T']:    vscale = 1.
                    if var_axis in ['Z3']:    vscale = 0.1    
                    if var_axis in ['PRECT']: vscale = 1000.
                    if var_axis in ['TAUX','TAUY']:  vscale = -1
                    if var_axis in ['OMEGA500','OMEGA']:  vscale = 36.   
                    if var_axis in ['Q']:  vscale = 1000.
                    if var_axis in ['DIV']:  vscale = 1.e+6  
                    if var_axis in ['CLOUD']:  vscale = 100.   
                    

                case 'TROPFLUX' if var_axis=='TAUX':
                    
                    da_axis = xr.open_dataset(dir_data+'tropflux/taux_tropflux_1m_1979-2018.nc')['taux']    
                    da_axis = da_axis.rename({'latitude': 'lat', 'longitude': 'lon'})
                    vscale = -1.

                case _:
                    
                    print("  - No obs, product match for ",case_type)



        case _ if case_type in ['cesm1','cesm2','cam5','cam6']:

            lens_chunk = {
                                "time": 12,     # one chunk per file (monthly)
                                "lev": -1,      # keep full vertical column
                                "lat": 64,      # or 48 / 72 depending on grid
                                "lon": 64}
            
            vscale = cvar_scale

            
            
            if case_type == 'cesm1':
                dir_lens = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/'
                fyrs_str = '.04*' if case == 'b.e11.B1850C5CN.f09_g16.005' else '*'
                
            if case_type == 'cesm2':
                dir_lens = '/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/month_1/'
                fyrs_str = '*'

            if case_type == 'cam5':
                dir_lens = '/glade/campaign/collections/cdg/data/CAM_prescribed_SST/f.e11.FAMIPC5CN.f09_f09.hist-rcp85.goga/atm/proc/tseries/monthly/'
                fyrs_str = '*'

            
            if case_type == 'cam6':
                dir_lens = '/glade/campaign/collections/gdex/data/d651010/global/CESM2.1_GOGA_ERSSTv5/atm/proc/tseries/month_1/'
                fyrs_str = '*'
                
        

            
                
            file_suff = var_axis+'/'+case+'.cam.h0.'+var_axis+fyrs_str+'.nc'
            files_hist = dir_lens+file_suff
 
      
           
            
            files_ls  = glob.glob(files_hist)         

      
            
            
            print('  - Grabbing file(s) for LENS1/2 (CESM1/2)')

#            da_axis = xr.open_mfdataset(files_ls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")[var_axis]


            if case_type in ['cesm1','cam5','cam6'] and var_axis == 'PRECT': # Need to grab PRECC and PRECL files separately 
    
                files_hist_pc = files_hist.replace('PRECT', 'PRECC')
                files_hist_pl = files_hist.replace('PRECT', 'PRECL')
    
                files_ls_pc  = glob.glob(files_hist_pc)
                files_ls_pl  = glob.glob(files_hist_pl)
                
                da_pc = xr.open_mfdataset(files_ls_pc,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",chunks=lens_chunk)['PRECC']
                da_pl = xr.open_mfdataset(files_ls_pl,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",chunks=lens_chunk)['PRECL']
                
                da_axis = da_pc+da_pl
                
            else:

                # If empty file then try my derived directory
                if not files_ls:
                    print('-Checking for local CESM copy, likely a derived variable if it exists')
                    files_hist = dir_ncout+file_suff
                    files_ls  = glob.glob(files_hist)
                   
                    

                # Now try a 3D derived data grab
                if not files_ls: # Either the variable just isn't there or we have to derive it from a 3D variable
                    
                    if lvar_from3d: # Test if the last 3 letters of the variable name are a digit (e.g., 200) to determine that it is a single level  
                        
                        files_suff = var_3dget+'/'+case+'.cam.h0.'+var_3dget+fyrs_str+'.nc'
                        files_suff = dir_lens+files_suff.replace(var_3dget, var_axis)
                        
                        print('-Detemining that we need to process ',var_3dget,' to determine ',var_axis)
                        
                        da_axis = cam_2d_from_3d(var_axis,var_3dget,var_plev,files_suff=files_suff)
                    else:

                        print('Variable is not anywhere - do not know what to do...')
                    

                else: # File(s) exist!

                    
                    ds_axis = xr.open_mfdataset(files_ls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",compat='override',chunks=lens_chunk)

                    
                    if "lev" in ds_axis[var_axis].dims:
                        file_psuff = 'PS/'+case+'.cam.h0.PS'+fyrs_str+'.nc'
                        files_phist = dir_lens+file_psuff
                        files_pls  = glob.glob(files_phist) 
                        ds_ps = xr.open_mfdataset(files_pls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",compat='override',chunks=lens_chunk)
                        
                        da_axis = get_plev_cam(ds_axis,ds_ps,grab_era5_levs(),var_axis)
                    else:
                        da_axis = ds_axis[var_axis] 
                
        
           
    
        case _ if case_type == 'cesm3':


            # Does a timeseries dataset exist for this case?
            # Check for this file name lwoer down    

            
            ofile_case = dir_ncout+case+'_'+var_axis+'_mmeans_ts.nc'

      
            # add vscale here
            vscale = cvar_scale
            
            if os.path.exists(ofile_case) and lread_ts_file:
    
                print('  - Timeseries files exist for the case - using them')
            
                print('     ',ofile_case)
            
                da_axis = xr.open_dataset(ofile_case)[var_axis]


            else:
                
                # Pick the right directory (hannay/gmarques)
    
                dir_c3 = '/glade/derecho/scratch/hannay/archive/'
                
                if os.path.isdir(dir_c3+case):
                    print("   - Cecile's Run")
                # Your operation here, e.g. read/write files
                else:
                    print("   - Gustavo's Run")
                    dir_c3 = '/glade/derecho/scratch/gmarques/archive/'
    
                
    # Grab files and read in.
    
                # Trim down range of files to read in requested years, otherwise read in all.
    
                
                if not lread_in_all_hist:
                    files_hist = []
                    yrange = list(range(yr0, yr1+1))
    #                yr_arr_string = "[" + ",".join(f"{n:04d}" for n in yrange) + "]"
                    yr_arr_strings = [f"{num:04d}" for num in yrange]
    
    #                print(y_arr_strings)
                    for yr_str in yr_arr_strings:
    #                    print(yr_str)
                        file_ls = dir_c3+case+'/atm/hist/'+case+'.cam.h0a.'+yr_str+'*.nc'
                        files_hist.extend(glob.glob(file_ls)) 
    
                    files_hist.sort()
                
                else:
                    
                    file_hist = dir_c3+case+'/atm/hist/'+case+'.cam.h0a.*.nc'
                    
               
     
     # File checks           
                
                print('  - Reading ',len(files_hist),' files  (first/last)')
                print('   -',files_hist[0])
                print('   -',files_hist[-1])
                
            # Open them as multiple files
    
                print('  - Slow read of h0a output...')

                # Grab the h0a datasets.
                ds_axis = xr.open_mfdataset(files_hist,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")


                if lvar_from3d: # Test if the last 3 letters of the variable name are a digit (e.g., 200) to determine that it is a single level                   
                    print('-Detemining that we need to process ',var_3dget,' to determine ',var_axis)
                        
                    da_axis = cam_2d_from_3d(var_axis,var_3dget,var_plev,ds_h0=ds)

                else:

                    da_axis = ds_axis[var_axis]
                
    
                print('  -Done')
                
            # Write out the timeseries of 2D field files (unscaled)?
            
                if lwrite_ts_file: 
    
                    print("  - Write out files of 2D field "+var_axis)
        
                    print('    ',ofile_case)
        
        
                    da_axis.to_netcdf(ofile_case,mode="w")
                    da_axis.close()

    
                    print(' - Done')
    
        case _:
            
            print('  - No case_type match for '+case)

# Squeeze out single value dimensions (usually pressure)
    da_axis = da_axis.squeeze()

    
# Just scale the variable right at the end

    da_axis = vscale *  da_axis

    

    return da_axis




    
