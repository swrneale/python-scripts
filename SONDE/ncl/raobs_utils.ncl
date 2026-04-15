load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"   

;----------------------------------------------------------------------
; This procedure reads and prints key information about this
; RAOB station
;
; You will likely need to change lat_values and/or lon_values to
; contain the locations where you want lat/lon labels.
;----------------------------------------------------------------------


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function isvar2d(varname)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
begin

; Retuns true iff returned variable is 2D

  vars3d = (/"U","V","T","Q"/)

  if (any(varname.eq.vars3d)) then
    l3d = True
  else
    l3d = False
  end if
      
  return (l3d)

end



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function raob_file(station_in,rdir_in)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin
; Which station number

  station_names = (/"Albuquerque","Churchill","Davenport","Denver","Egedesminde","KeyWest","LakeCharles","Minneapolis","Narassarssuaq","Norman","OklahomaCity","Omaha","SantaTeresa","Slidell","Springfield","Tallahassee","Thule","Topeka","Tucson","Valparaiso","Kingston","EasterIsland","SanCristobal","Hilo","Tarawa","GrandCayman"/)

; Denver has occasional 1000ish values.

  rfile_in = rdir_in+"raob_soundings_"+station_in+".cdf"

  print(" ")
  print(" ")
  print(" ")

  if (any(station_in.eq.station_names)) then
    print(" ")
    print("******************************************************")
    print(" Station selected: "+station_in+" is available")
    print("******************************************************")
  else
    print(" ")
    print("******************************************************")
    print(" Station : "+station_in+" is not available. EXITING...")
    print("******************************************************")
    exit
  end if

  if (fileexists(rfile_in)) then
    print("-"+rfile_in+" : Exists")
  else
    print("-"+rfile_in+" : Does not exist ..... EXITING")
    exit
  end if
 

  return(rfile_in)

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function cam_file(crun_name,cdir_in,cvarname,res)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin
; CAM file: exits and info?

  print(" ")
  print(" ")

  print("******************************************************")
  print(" CAM simulation selected is : "+ crun_name)
  print("******************************************************")

; Does single processed file exist in CAM directory? 

  fstub = cvarname+"/"+crun_name+".cam."+res@chan+"."+cvarname
  pfile_in = cdir_in+fstub+"."+res@station+".nc"
 
  if (fileexists(pfile_in)) then
    print("- Processed CAM file exists for : "+cvarname)
    print("File: "+pfile_in)
    res@lcproc = False
    cfile_in = pfile_in
  else
    print("- Processed CAM file does not exist for : "+cvarname)
    print("- Will require processing (slow)!")
    res@lcproc = True
    if (any(cvarname.eq.(/"U","V","T","Q","DTCOND","PS","PRECT"/))) then ; Need my own copies as not on disk in p/cesm
;       cdir_raw = "/glade/campaign/cgd/amp/rneale/cam6_revert/"
      cdir_raw = "/glade/campaign/cgd/amp/rneale/large_ens/"
    else
      cdir_raw = "/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/hourly6/"
    end if
    cfile_in = systemfunc("ls -1 "+cdir_raw+fstub+"*.nc")
    print("File(s): "+cfile_in)

; Iff we have a 2d variable that hasn't been created for this location thenn we need PS to regrid.
      if (isvar2d(cvarname)) then ; Attach file location to res.
        cdir_raw = "/glade/work/rneale/large_ens/"
;        cdir_raw = "/glade/work/rneale/cam6_revert/"
        ps_stub = "PS/"+crun_name+".cam."+res@chan+".PS"
        res@ps_file = systemfunc("ls -1 "+cdir_raw+ps_stub+"."+res@station+".nc") 
        print("PS file(s): "+res@ps_file)
      end if

  end if
  
  res@pfile = pfile_in
  res@pdir = cdir_in

  print(" ")


  return(cfile_in)

end





;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function get_raob_times(fin)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

; Time info, a little complex
  print("** RAOB: Grabbing time coordinate...")
  print(" ")
  rtime = fin->synTime

; Not sure below will always be universally true.
  if  (rtime@units.eq."Seconds since (1970-1-1 00:00:0.0)") then
    rtime@units = "Seconds since 1970-1-1 00:00:0.0"
  end if

; No missing value and need to ensure we get correct one for ncl.

  rtime@_FillValue = default_fillvalue(typeof(rtime))
  rtime!0 = "time"



  return(rtime)

end



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function get_raob_vname(fin,ltype,name_in)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin
; Grab data from raqobs file for a variable.
; Tried to list variables on maondatory levels where the first level is the surface

  cnames = (/"PS","T","U","V","VMAG","Q","DPD","RELHUM"/)
  
  if (ltype.eq."M") then
    rnames = (/"prMan","tpMan","U","V","wsMan","Q","tdMan","RELHUM"/) ; Sometimes it will be the CAM name iff more than one variable is needed.
  end if

  if (ltype.eq."T") then
    rnames = (/"prSigT","tpSigT","U","V","wsSigT","Q","tdSigT","RELHUM"/) ; Sometimes it will be the CAM name iff more than one variable is needed.
  end if

  if (ltype.eq."W") then
    rnames = (/"prSigT","tpSigW","U","V","wsSigW","Q","tdSigW","RELHUM"/) ; Sometimes it will be the CAM name iff more than one variable is needed.
  end if


  do ir=0,dimsizes(cnames)-1
    if (name_in.eq.cnames(ir)) then
      rname_out = rnames(ir)
    end if
  end do
  
  if (.not.isdefined("rname_out")) then
    print("** get_raob_vname: raob variable not defined for "+name_in+" EXITING...")
    exit
  end if

  return(rname_out)

end  ; get_raob_vname

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function raob_read(fin,cname,time_in,rdata)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin
; Grab data from raobs file for a variable.

  lev_type = rdata@lev_type

  rname = get_raob_vname(fin,lev_type,cname)

  print(" ")
  print("******************************************************")
  print(" Grabbing "+cname+" ("+rname+")")
  print("******************************************************")

  pi = 3.1415

; Zonal/meridional wind
  if (any(cname.eq.(/"U","V"/))) then ; Derive wind components from speed+direction.
    vin = (/"wsMan","wdMan"/)
    if (lev_type.eq."T") then
      vin = (/"wsSigT","wdSigT"/)
    end if
    if (lev_type.eq."W") then
      vin = (/"wsSigW","wdSigW"/)
    end if
    
    var_mag = fin->$vin(0)$
    var_dir = fin->$vin(1)$
    var_mag@_FillValue = default_fillvalue(typeof(var_mag)) ; Attach fill value for next calculation.
    var_dir@_FillValue = default_fillvalue(typeof(var_dir))
    
    var_out = var_mag
    
    if (cname.eq."U") then
      var_out = -var_mag*sin(var_dir*pi/180.)
      var_out@long_name = "Zonal Wind"
    end if
    if (cname.eq."V") then
      var_out = -var_mag*cos(var_dir*pi/180.)
      var_out@long_name = "Meridional Wind"
    end if
    var_out@units = "m/s"
    
    delete([/var_mag,var_dir,vin/])
    
  end if ; U,V calc.


; Specific humditiy.
    
  if (cname.eq."Q") then
      
; Select data level type.
    vin = (/"tdMan","tpMan","prMan"/)
    if (lev_type.eq."T") then
      vin = (/"tdSigT","tpSigT","prSigT"/)
    end if
    if (lev_type.eq."W") then
      vin = (/"tdSigW","tpSigW","prSigW"/)  
    end if
    
; Calculate from dew point depression      
    
    dpd = fin->$vin(0)$ ; Dew point dep.in K
    temp = fin->$vin(1)$ ; Temp. in K
    pres = fin->$vin(2)$ ; Pressure


    dpd@_FillValue = default_fillvalue(typeof(dpd)) ; Attach fill value for next calculation.
    temp@_FillValue = default_fillvalue(typeof(temp))
    pres@_FillValue = default_fillvalue(typeof(pres)) ; Attach fill value for next calculation.
    
    pres = (/pres*100./) ; -> Pa
    
    dpt = temp-dpd    ; Dew point temperature
    
    var_out = dpd
    var_out = mixhum_ptd (pres, dpt, -2) ; Specific humidity in g/kg       
    
    var_out@long_name = "Specific Humidity"
    var_out@units = "g/kg"

    delete([/temp,dpd,pres,vin/])
    
  end if ; Q calc.

; Relative humidity, need

  if (cname.eq."RELHUM") then

    vin = (/"tdMan","tpMan"/)
    if (lev_type.eq."T") then
      vin = (/"tdSigT","tpSigT"/)
    end if
    if (lev_type.eq."W") then
      vin = (/"tdSigW","tpSigW"/)  
    end if

    dpd = fin->$vin(0)$ ; Dew point dep.in K
    temp = fin->$vin(1)$ ; Temp. in K

; Fix fill values.
    dpd@_FillValue = default_fillvalue(typeof(dpd)) ; Attach fill value for next calculation.
    temp@_FillValue = default_fillvalue(typeof(temp))
   
    dpt = dpd
    dpt = (/temp-dpd/)

;    var_out = dpd
    printMinMax(dpd,True)
    printMinMax(temp,True)
    var_out = relhum_ttd (temp, dpt, 0)

    var_out@long_name = "Relative Humidity"
    var_out@units = "%"

    delete([/temp,dpd,dpt,vin/])

  end if


  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Deal with data based on whether significant or mandatory levels.  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; 1D vs. 2D data.

  if (cname.eq."PS") then
    print("-Reading..."+rname)
    var_out = fin->$rname$(:,0)
   
    print("-Done...")  
    var_out@units = "mb"
    var_out@long_name = "Surface Pressure"
  else    
   
; Read in file iff not a drived variable.

    if (.not.isdefined("var_out")) then
      print("-Reading..."+rname)
      var_out =  fin->$rname$ ; Read in variable.
      print("-Done...")  
    end if
;     
; Set up common output info.
    
    mplevs = rdata@man_plevels
    nmplevs = dimsizes(mplevs)

;
    if (any(lev_type.eq.(/"T","U"/))) then ; Sig level interpolation.
      
      if (lev_type.eq."T") then ; T or W sig/ levels?
        splevs = fin->prSigT ; Pressure profiles
      else
        splevs = fin->prSigQ
      end if
      
      var_out = raob_sig2p(var_out,splevs,mplevs) ; Interpolate from sig to pres. levels.
            
    else ; Mandatory levels.
      
; Trim off bottom layer as it is PS thenn mandatory layers above.
; Assumes continuous level selection above PS to nmplevs     
        

      var_tmp = var_out(:,1:nmplevs)
      var_tmp!1 = "lev"
      var_tmp&lev = mplevs
      var_tmp&lev@units = "mb"
      copy_VarCoords_1(var_out,var_tmp)
      copy_VarAtts(var_out,var_tmp)
      delete(var_out)
      var_out = var_tmp
      delete(var_tmp)
      
    end if ; sig T/u lev for processing.    

  end if ;1D(PS)/2D vars.

; Finally add time info.

  var_out!0 = "time"
  var_out&time = time_in

; Iff the time values themselves have missing values trim those timesfrom the variable.

  itime_good  = ind(.not.ismissing(var_out&time))

  if (dimsizes(dimsizes(var_out)).eq.1) then   
    var_out := var_out(itime_good)
  else
    var_out := var_out(itime_good,:)
  end if


; Should have correct missing value.

  var_out@_FillValue = default_fillvalue(typeof(var_out))
  var_out@missing_value = default_fillvalue(typeof(var_out))


; Unfortunate station/variable specific stuff

  if (cname.eq."PS") then
    
    if (any(rdata@station.eq.(/"Albuquerque","Tucson","Denver","Minneapolis","SantaTeresa"/))) then ; Get rid of 1000mb values (stupid missing values)
      if (any(var_out.eq.1000.)) then
        var_out(ind(var_out.eq.1000.)) = var_out@_FillValue
      end if
      if (any(var_out.eq.700.)) then
        var_out(ind(var_out.eq.700.)) = var_out@_FillValue
      end if
      if (any(var_out.eq.819.)) then
        var_out(ind(var_out.eq.819.)) = var_out@_FillValue
      end if

    end if
     
  end if


; Attach station info to be used for plotting titles later.
  var_out@stname = rdata@station
  var_out@lat = rdata@lat
  var_out@lon = rdata@lon
  var_out@elev = rdata@elev
  var_out@cname = cname

  return(var_out)

end ; raob_read







;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function cam_read(fin,cname,rdata)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin
; Grab CAM variable from file. Should be easier than raobs.
; Also iff needed extract out the single point nearest the raobs location.
; We should probably doo a proper calculation of fields interpolated to
;  the same standrad pressure levels as raobs.

  print(" ")
  print("******************************************************")
  print(" CAM: Grabbing "+cname)
  print("******************************************************")

  slat = rdata@lat 
  slon = rdata@lon
  
; Station details.
  print("-- RAOB: Station location is : "+slat+"N, "+slon+"E")

; Processed file exists?

  lcproc = rdata@lcproc
  pfile = rdata@pfile
  pdir = rdata@pdir
  
; Grab raw data for processing..
    
  if (lcproc) then
    
    print("- Extracting nearest grid-point from raw file")
    
; Read file lat/lon to determine single grid-point to extract.
    
    clat_in = fin->lat
    clon_in = fin->lon
    
; Trim off the single point.
    
    nclon = dimsizes(clon_in)
    nclat = dimsizes(clat_in)
    
; Generate 2d lat and lon arrays
    
    clon2d = new((/nclat,nclon/),double)
    clat2d = clon2d
    
    do ix=0,nclon-1
      clat2d(:,ix) = clat_in
    end do
    do iy=0,nclat-1
      clon2d(iy,:) = clon_in
    end do
    
; Grab nearest lat/lon
    
    nloc_ind = getind_latlon2d(clat2d, clon2d, slat, slon)
    lat_gp = clat2d(nloc_ind(0,0),nloc_ind(0,1))
    lon_gp = clon2d(nloc_ind(0,0),nloc_ind(0,1))
    ilat_gp = nloc_ind(0,0)
    ilon_gp = nloc_ind(0,1)

    print("- Nearest CAM grid point is : "+sprintf("%6.2f",lat_gp)+"N, "+(360.-lon_gp)+"E")
    print("- Extracting grid-point data...")
    
; Now grab the variable since we know it's lat/lon location.
    
    if (dimsizes(getfilevardims(fin,cname)).eq.4) then
      print("- Vertical interpolation using CAM PS file.")
; 2d variable: Need to intepolate to "mandatory" raob pressure levels.
      
      var_in = fin->$cname$(:,:,{lat_gp},{lon_gp})
      dims = dimsizes(var_in)
      
; Process. No extrapolation and unfortunately adding lat/lon dim =1 to perform vertical intepolation annoying.

      fps = addfile(rdata@ps_file,"r")

      ps = fps->PS
      nt = dims(0)
      np = dims(1)
      ps_ll = reshape(ps,(/nt,1,1/))
      var_in_ll = reshape(var_in,(/nt,np,1,1/))
        
; Grabs hybrid coord info from raw CAM file(S). Remove lat/lon order 1 dimensions.

      var_out = vinth2p(var_in_ll,fin->hyam,fin->hybm,rdata@man_plevels,100.*ps_ll,2,0.01*(fin->P0),1,True)
      var_out := var_out(:,:,0,0)
      
; Set up lev dimension
      copy_VarCoords_1(var_in, var_out) ; Copy coords for time.
      var_out!1 = "lev"
      var_out&lev = rdata@man_plevels
      var_out&lev@units = "mb"
        
; Add variable meta data.
      copy_VarAtts(var_in,var_out)

; 1-d processing
    else ; 1d var: no more processing required.
      var_out = fin->$cname$(:,{lat_gp},{lon_gp})
    end if
    
    print("- Done ...")
    
; Write out variable and some attributes (after maybe scaling stuff)
    
    if (cname.eq."PS") then
      var_out = (/var_out*0.01/) ; pa -> mb
      var_out@units = "mb"
      var_out@long_name = "Surface Pressure"
    end if

    if (cname.eq."PRECT") then
      var_out = (/var_out*86400.*1000./) ; pa -> mb
      var_out@units = "mm/day"
      var_out@long_name = "Total Precipitation"
    end if

    if (cname.eq."DTCOND") then
      var_out = (/var_out*86400./) ; pa -> mb
      var_out@units = "K/day"
      var_out@long_name = "Moist Physics dT/dt"
    end if
    
    if (cname.eq."Q") then
      var_out = (/var_out*1000./) ; pa -> mb
      var_out@units = "g/kg"
      var_out@long_name = "Specific Humidity"
    end if

    yyyymmdd = cd_calendar(var_out&time,-2)
    yyyymmdd!0 = "time"
    yyyymmdd&time = var_out&time

    print("- Write out grid-point, time-series file") 
    system("mkdir "+pdir+cname+" >>/dev/null")
    fp = addfile(pfile,"c") ; Open CAM proc file for one grid-point.
    filedimdef(fp,"time",-1,True)
    
    fp->$cname$ = var_out
    fp->date = yyyymmdd
    fp@date_written = systemfunc("date")
    fp@station = rdata@station
    fp@station_id = rdata@stwid
    fp@lat = lat_gp
    fp@lon = lon_gp
    
    print("- Done ...")
      
  else

; Otherwise just grab the data of the processed file with a single grid point.

    print("- Grabbing variable from already processed file.")
    
    var_out = fin->$cname$
      
  end if
    
; Time information.
    
  date =  cd_calendar(var_out&time,0)
  ntimes = dimsizes(date(:,0))
  
  year0  =  toint(date(0,0))
  year1  =  toint(date(ntimes-1,0))
  month0 =  toint(date(0,1))
  month1 =  toint(date(ntimes-1,1))
  day0   =  toint(date(0,2))
  day1   =  toint(date(ntimes-1,2))
  hour0  =  toint(date(0,3))
  hour1  =  toint(date(ntimes-1,3))
  
  month_abbr = (/"","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep", \
  "Oct","Nov","Dec"/) 
  
  print(" ")
  print("*****************************************************************")
  print("TIME INFO FROM CAM FILE")
  print("*****************************************************************")
  print("Time range = "+sprinti("%0.2iZ ", hour0) + sprinti("%0.2i ", day0) + month_abbr(month0) + " "  + sprinti("%0.4i", year0)+" to "+ \
  sprinti("%0.2iZ ", hour1) + sprinti("%0.2i ", day1) + month_abbr(month1) + " "  + sprinti("%0.4i", year1))

  if (rdata@year_first.lt.year0.or.rdata@year_last.gt.year1) then
    print("! WARNING !: Requested year range of "+rdata@year_first+"-"+rdata@year_last+" lies outside available range of "+year0+"-"+year1)
  end if
 


; Should have correct missing value.

  var_out@_FillValue = default_fillvalue(typeof(var_out))

 
;;; Adding some info.
    
; CAM elevation: Generalized to eventually pick up off topo file and add to generated files above.
  
  f_lar_ens = addfile("/glade/campaign/cgd/amp/rneale/large_ens/V/b.e11.B20TRC5CNBDRD.f09_g16.001.cam.h2.V.1990010100Z-2005123118Z.nc","r")
  topo_file = f_lar_ens@topography_file
  ftopo = addfile(topo_file,"r")
  var_out@cam_elev = 0.1*(ftopo->PHIS({var_out@lat},{var_out@lon}))

; Other stuff  

  var_out@stname = rdata@station
  var_out@lon = var_out@lon-360. ; Switch lons to match raobs.
  var_out@elev = rdata@elev
  var_out@cname = cname
  
  return(var_out)

end ; cam_read 






;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure raob_plot_res_1d(res)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

;;;;; Plot 1d raob data


res@gsnMaximize         = True                   ; maximize plot in frame
res@xyMarkLineModes     = "Markers"  ; choose which have markers
res@xyMarkers           = 1                     ; choose type of marker 
res@xyMarkerColor       = "black"                  ; Marker color
res@xyMarkerSizeF       = 0.001                  ; Marker size (default 0.01)
res@xyDashPatterns      = (/0,0,0/)                      ; solid line 
res@xyMonoLineThickness = False
res@xyLineThicknesses   = (/3,3,3,3,3/)               ; set second line to 2

;Mapping monthly names onto doy.

res@tmXBMode = "Explicit"
res@tmXBLabelFontHeightF = 0.015
;res@tmXBMinorOn = True
;res@tmXBMajorOn = False
res@tmXBMajorLengthF = 0.
res@tmYLMajorLengthF = 0.01
res@tmXBLabelDeltaF = -1.

res@tmXMinorGrid = True ; Vertical lines
res@tmXMinorGridThicknessF = 1.
res@tmXMinorGridLineDashPattern = 14.


res@tmXBValues =  -15.5+(/31,61,92,122,153,183,214,245,275,305,333,366/)
res@tmXBMinorValues = (/1,32,61,92,122,153,183,214,245,275,305,333,366/)
res@tmXBLabels = (/"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)

res@trXMinF  = 1
res@trXMaxF  = 366


res@tiXAxisString = "Month"

end


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure raob_plot_res_tp(res)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

;;;;; Resources to plot 1d raob data

cname = res@cname ; Variable name needed for contour choices.
delete(res@cname)
res@gsnMaximize         = True                   ; maximize plot in frame


;Mapping monthly names onto doy (x-axis stuff)

res@tmXBMode = "Explicit"
res@tmXBLabelFontHeightF = 0.015

res@tmXBMajorLengthF = 0.
res@tmYLMajorLengthF = 0.01
res@tmXBLabelDeltaF = -1.

res@tmXMinorGrid = True ; Vertical lines
res@tmXMinorGridThicknessF = 1.
res@tmXMinorGridLineDashPattern = 14.


res@tmXBValues =  -15.5+(/31,61,92,122,153,183,214,245,275,305,333,366/)
res@tmXBMinorValues = (/1,32,61,92,122,153,183,214,245,275,305,333,366/)
res@tmXBLabels = (/"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)

res@trXMinF  = 1
res@trXMaxF  = 366 

res@tiXAxisString = "Month"

; y-axis stuff


; Master contour stuff

res@cnLevelSelectionMode = "ManualLevels"    
res@cnFillOn             = True                  ; turn on color
res@cnLinesOn            = True                 ; no contour lines
res@cnInfoLabelOn        = False
res@cnLineLabelFontHeightF = 0.01

res@gsnContourNegLineDashPattern = 14 
res@gsnContourZeroLineThicknessF = 3

; Variable dependent stuff

if (any(cname.eq.(/"T"/))) then
  if (res@lp2d_nomean) then
    res@cnMinLevelValF       =  -5.                  ; set minn contour level
    res@cnMaxLevelValF       =  5.                  ; set maxx contour level
    res@cnLevelSpacingF      =  0.5                  ; set contour spacing
  else
    res@cnMinLevelValF       =  220.                  ; set minn contour level
    res@cnMaxLevelValF       =  300.                  ; set maxx contour level
    res@cnLevelSpacingF      =  5.                  ; set contour spacing
  end if
end if

if (any(cname.eq.(/"DTCOND"/))) then
  if (res@lp2d_nomean) then
    res@cnMinLevelValF       =  -10.                  ; set minn contour level
    res@cnMaxLevelValF       =  10.                  ; set maxx contour level
    res@cnLevelSpacingF      =  1.                ; set contour spacing
  else
    res@cnMinLevelValF       =  -10.                  ; set minn contour level
    res@cnMaxLevelValF       =  10.                  ; set maxx contour level
    res@cnLevelSpacingF      =  1.                  ; set contour spacing
  end if
end if


if (any(cname.eq.(/"Q"/))) then

  if (res@lp2d_nomean) then
    res@cnMinLevelValF       =  -5.                  ; set minn contour level
    res@cnMaxLevelValF       =   5.                  ; set maxx contour level
    res@cnLevelSpacingF      =   0.5                 ; set contour spacing
  else
    res@cnMinLevelValF       =  0.5                 ; set minn contour level
    res@cnMaxLevelValF       =  20.                  ; set maxx contour level
    res@cnLevelSpacingF      =  0.5                  ; set contour spacing
  end if


end if

if (any(cname.eq.(/"VMAG","U"/))) then
  res@cnMinLevelValF       =  -10.                  ; set minn contour level
  res@cnMaxLevelValF       =  10.                  ; set maxx contour level
  res@cnLevelSpacingF      =  0.5                  ; set contour spacing
end if

if (any(cname.eq.(/"V","DPD"/))) then
  res@cnMinLevelValF       =  -6.                  ; set minn contour level
  res@cnMaxLevelValF       =  6.                  ; set maxx contour level
  res@cnLevelSpacingF      =  0.5                  ; set contour spacing
end if

if (any(cname.eq.(/"RELHUM"/))) then
  res@cnMinLevelValF       =  10.                  ; set minn contour level
  res@cnMaxLevelValF       =  100.                  ; set maxx contour level
  res@cnLevelSpacingF      =  5.                  ; set contour spacing
end if

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure raob_info(station_in,fr:file,rtime,rdata_out)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

; Raobs file station info.

  rdata_out@lat   = fr->staLat(0)
  rdata_out@lon   = fr->staLon(0)
  rdata_out@elev  = fr->staElev(0)
  rdata_out@stwid = fr->wmoStat(0)

 ; Manditory levels.
  if (rdata_out@lev_type.eq."M") then
    rdata_out@man_plevels = (/1000., 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10/)
  else
; Level increments for significant level plotting (wind or temp)
    dplev = 50.
    pmax = 1000.
    pmin = 50.
    nplevs = toint((pmax-pmin)/dplev)
    rdata_out@man_plevels = fspan(1000,10,nplevs+1)
  end if
  print("-Output levels selected -- "+rdata_out@man_plevels+" mb")

; Set date for verbose printing. Using ut_cal since cd_cal produces errors.
; Hoping it's not important

  date = cd_calendar(rtime,0)
  ntimes = dimsizes(date(:,0))

  year0  =  toint(date(0,0))
  year1  =  toint(date(ntimes-1,0))
  month0 =  toint(date(0,1))
  month1 =  toint(date(ntimes-1,1))
  day0   =  toint(date(0,2))
  day1   =  toint(date(ntimes-1,2))
  hour0  =  toint(date(0,3))
  hour1  =  toint(date(ntimes-1,3))

  month_abbr = (/"","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep", \
                    "Oct","Nov","Dec"/) 

; 
  print(" ")
  print("*****************************************************************")
  print("INFO FROM RAOB FILE")
  print("*****************************************************************")
  print("Station: "+station_in+" (WMO identifier: "+rdata_out@stwid+")")
  print("Latitude: "+rdata_out@lat+" Longitude: "+rdata_out@lon+" elevation: "+rdata_out@elev+"m") 
  print("Time range = "+sprinti("%0.2iZ ", hour0) + sprinti("%0.2i ", day0) + month_abbr(month0) + " "  + sprinti("%0.4i", year0)+" to "+ \
                        sprinti("%0.2iZ ", hour1) + sprinti("%0.2i ", day1) + month_abbr(month1) + " "  + sprinti("%0.4i", year1))


end



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function get_obs_precip(pdset,rdata)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

; Function to get rainfall as a 1D data structure from the nearest grid-point of the data set.

;;; Data-sets
; /CPC?

  if (pdset.eq."CPC") then
    pfile = "/glade/work/rneale/data/NOAA_CPC_USA/precip.V1.0.day.ltm.nc"
    fin = addfile(pfile,"r")
    pname = "precip"
  end if

  if (pdset.eq."TRMM") then
    pfile = "/glade/work/rneale/data/TRMM/3B42.1998-2009.nc"
    fin = addfile(pfile,"r")
    pname = "precip"
  end if

  if (pdset.eq."GPCP") then
    pfile = "/glade/work/rneale/data/GPCP/GPCP_1DD_v1.2_199610-201407.nc"
    fin = addfile(pfile,"r")
    pname = "PREC"
  end if

  if (pdset.eq."Livneh") then
    pfile = "/glade/work/rneale/data/Livneh/prec.day.ltm.nc"
    fin = addfile(pfile,"r")
    pname = "prec"
  end if

  if (pdset.eq."PRISM") then
    pfile = "/glade/work/rneale/data/PRISM/daily/PRISM_day_1981-2014.nc"
    fin = addfile(pfile,"r")
    pname = "PRECT"
  end if

; Find nearest grid-point (using 2D lat/lon arrays)

  slat = rdata@lat
  slon = rdata@lon

; Read file lat/lon to determine single grid-point to extract.
  
  clat_in = fin->lat
  clon_in = fin->lon
  
  nclon = dimsizes(clon_in)
  nclat = dimsizes(clat_in)
  
; Generate 2d lat and lon arrays
  
  clon2d = new((/nclat,nclon/),double)
  clat2d = clon2d
  
  do ix=0,nclon-1
    clat2d(:,ix) = clat_in
  end do
  do iy=0,nclat-1
    clon2d(iy,:) = clon_in
  end do
  
  nloc_ind = getind_latlon2d(clat2d, clon2d, slat, slon)
  lat_gp = clat2d(nloc_ind(0,0),nloc_ind(0,1))
  lon_gp = clon2d(nloc_ind(0,0),nloc_ind(0,1))
  ilat_gp = nloc_ind(0,0)
  ilon_gp = nloc_ind(0,1)
  
  print("- Nearest "+pdset+" grid point is : "+sprintf("%6.2f",lat_gp)+"N, "+(360.-lon_gp)+"E")
  print("- Extracting grid-point data...")

; Extract grid-point data.
  
  var_in = fin->$pname$(:,{lat_gp},{lon_gp})

  return(var_in)

end ; get_obs_precip



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure station_1d_plot(wks,vplot_in,rdata)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

;;;;; Plot 1d raob data

;1. Grab date info for plotting  
  
date = toint(cd_calendar(vplot_in&time,0))

years = date(:,0)
year0 = rdata@year_first
year1 = rdata@year_last
drunave = rdata@drunave


;print(vplot_in&time)

; Day of year for plot
;print(vplot_in&time(441))
;print(date(442,:))


doy = day_of_year(date(:,0),date(:,1),date(:,2))
idoy_good  = ind(.not.ismissing(doy))

doy := doy(idoy_good)
vplot_in := vplot_in(idoy_good)




;print(doy@_FillValue)
;print(date(0:100,0)+" "+date(0:100,1)+" "+date(0:100,2))
ndoys = max(doy)

res                     = True                   ; plot mods desired
res@tiMainString        = "Scatter plot"  ; title

; Lets grab some common resources we shouldn't repeat for the time axis.

raob_plot_res_1d(res)

;Local resources.
res@tiMainString        = vplot_in@long_name
res@tiYAxisString       = vplot_in@units

res@tiMainString = rdata@station+" ("+rdata@lat+" lat, "+rdata@lon+" lon, "+rdata@elev+"m elev.)"

res@gsnDraw = False
res@gsnFrame = False

res@trYMaxF = 1006.
res@trYMinF = 1022.

;res@trYMinF = min(vplot_in)
;res@trYMaxF = max(vplot_in)

; Text
txres = True
txres@txFontHeightF = 0.02  

; Initial scatter plot of alll sondes.
plot  = gsn_csm_xy (wks,doy,vplot_in,res)

; Day of year array (w/ leap) 
ac_doy = fspan(1,ndoys,ndoys)

; Switch to lines.
res@xyMarkLineModes     = "Lines"

; Trim for years (maskk out will _Fill)

vplot_in = where(years.ge.year0.and.years.le.year1,vplot_in,vplot_in@_FillValue)
;print(years)
;print(doy)
if (rdata@lp1d_av_std) then ; Plot +/- 1 std.
  ac_plot = new((/3,ndoys/),"float")
  ac_plot = default_fillvalue(typeof(ac_plot))  
  ac_plot!1 ="doy"
 
  do it=0,ndoys-1
    
    id = it+1
    doy_ave = doy.eq.id

    idoy_ave := ind(doy.eq.id)

; Iff no days exist for thsi averaging doy thenn doo not create and ave.

    if (any(doy_ave)) then ; No days for this doy? Thenn skip averaging
      
;      print(id+" "+any(doy_ave))
;      print(""+idoy_ave+" "+dimsizes(idoy_ave))

;      print(""+vplot_in(ind(doy_ave)))
      ac_plot(0,it) = dim_avg(vplot_in(ind(doy_ave))) ; mean
      
      
      ac_plot(1,it) = ac_plot(0,it)-dim_stddev(vplot_in(idoy_ave)) ; -1 std
      ac_plot(2,it) = ac_plot(0,it)+dim_stddev(vplot_in(idoy_ave)) ; +1 std
    else
      print("-Warning no days exist for doy = "+(id+1))
    end if
  end do

; Attach +/- one std for each day.  


  plot1 = gsn_csm_xy (wks,ac_doy,runave(ac_plot,drunave,-1),res)
;  plot1 = gsn_csm_xy (wks,ac_doy,ac_plot,res)
  overlay(plot,plot1)
  delete(ac_plot)
end if

if (rdata@lp1d_0012Z) then ; Plot 00Z and 12Z separately.
  if (rdata@cam) then
    ac_plot = new((/4,ndoys/),"float")
  else
    ac_plot = new((/2,ndoys/),"float")
  end if
  zhr = date(:,3)

  do it=0,ndoys-1
;    print(ind(doy.eq.id.and.zhr.eq.0))
    id = it+1
    ac_plot(0,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.0)))
    ac_plot(1,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.12)))
    if (rdata@cam) then
      ac_plot(2,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.6))) 
      ac_plot(3,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.18)))
    end if
  end do


  res@xyLineColors = (/"blue","red","darkgreen","brown"/)

  plot1 = gsn_csm_xy (wks,ac_doy,runave(ac_plot,drunave,-1),res)
  overlay(plot,plot1)

; Put a little text on for a key.
  
  xtext = 0.95*max(ac_doy)

  ytext = 0.86*(res@trYMaxF-res@trYMinF)+res@trYMinF
  txres@txFontColor =  res@xyLineColors(0)
  dum0 = gsn_add_text(wks,plot,"00Z",xtext,ytext,txres) 

  ytext = 0.94*(res@trYMaxF-res@trYMinF)+res@trYMinF
  txres@txFontColor =  res@xyLineColors(1)
  dum0 = gsn_add_text(wks,plot,"12Z",xtext,ytext,txres) 
  
  if (rdata@cam) then  
    ytext = 0.90*(res@trYMaxF-res@trYMinF)+res@trYMinF
    txres@txFontColor =  res@xyLineColors(2)
    dum1 = gsn_add_text(wks,plot,"06Z",xtext,ytext,txres) 
  
    ytext = 0.98*(res@trYMaxF-res@trYMinF)+res@trYMinF
    txres@txFontColor =  res@xyLineColors(3)
    dum1 = gsn_add_text(wks,plot,"18Z",xtext,ytext,txres) 
  end if

  delete(ac_plot)

end if

;
; Last thing is to printt the CAM lat/lon on bottomleft of plot
; For sanity

if (rdata@cam) then
  txres@txFontHeightF = 0.015
  xtext = 0.28*max(ac_doy)
  ytext = 0.03*(res@trYMaxF-res@trYMinF)+res@trYMinF

  txres@txFontColor =  1
  dumc = gsn_add_text(wks,plot,"CAM grid-point: "+sprintf("%5.2f",vplot_in@lat)+"N, "+sprintf("%5.2f",(vplot_in@lon))+"E, "+sprintf("%4.0f",vplot_in@cam_elev)+"m",xtext,ytext,txres) 
end if

draw(plot)


end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure station_precip_plot(wks,vplot_in,rdata)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

;;;;; Plot 1d rainfall data
; Assume CAM data passed on entrance.

;1. Grab date info for plotting  
  
date = toint(cd_calendar(vplot_in&time,0))
years = date(:,0)
year0 = rdata@year_first
year1 = rdata@year_last
drunave = rdata@drunave

; Day of year for plot
doy = day_of_year(date(:,0),date(:,1),date(:,2))
ndoys = max(doy)

res                     = True                   ; plot mods desired
res@tiMainString        = "Scatter plot"  ; title

; Lets grab some common resources we shouldn't repeat for the time axis.

raob_plot_res_1d(res) ; To create.

;Local resources.
res@tiMainString        = vplot_in@long_name
res@tiYAxisString       = vplot_in@units

res@tiMainString = rdata@station+" ("+rdata@lat+" lat, "+rdata@lon+" lon, "+rdata@elev+"m elev.)"

res@gsnDraw = False
res@gsnFrame = False

res@trYMinF = 0.
res@trYMaxF = 10.

;res@trYMinF = min(vplot_in)
;res@trYMaxF = max(vplot_in)

; Text
txres = True
txres@txFontHeightF = 0.025  

; Initial scatter plot of alll sondes.
;plot  = gsn_csm_xy (wks,doy,vplot_in,res)

; Day of year array (w/ leap) 
ac_doy = fspan(1,ndoys,ndoys)

; Switch to lines.
res@xyMarkLineModes     = "Lines"

; Trim for years (maskk out will _Fill)

vplot_in = where(years.ge.year0.and.years.le.year1,vplot_in,vplot_in@_FillValue)


if (rdata@lp1d_av_std) then ; Plot +/- 1 std.
  ac_plot = new((/3,ndoys/),"float")
  ac_plot = 0.
  ac_plot!1 ="doy"
 
  do it=0,ndoys-1
    id = it+1
    ac_plot(0,it) = dim_avg(vplot_in(ind(doy.eq.id))) ; mean
    ac_plot(1,it) = ac_plot(0,it)-dim_stddev(vplot_in(ind(doy.eq.id))) ; -1 std
    ac_plot(2,it) = ac_plot(0,it)+dim_stddev(vplot_in(ind(doy.eq.id))) ; +1 std
  end do

; Attach +/- one std for each day.  

  plotc = gsn_csm_xy (wks,ac_doy,runave(ac_plot(0,:),drunave,-1),res)
;  overlay(plotc,plot1)
  delete(ac_plot)
end if

;;;;;;;;;

if (rdata@lp1d_0012Z) then ; Plot model 00Z and 12Z separately.
  hrs = (/"00","06","12","18"/)
  nlines = dimsizes(hrs)+1
  ac_plot = new((/nlines,ndoys/),"float")
  zhr = date(:,3)
  do it=0,ndoys-1
    id = it+1
    ac_plot(0,it) = dim_avg(vplot_in(ind(doy.eq.id))) ; mean
    ac_plot(1,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.00))) 
    ac_plot(2,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.06))) 
    ac_plot(3,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.12))) 
    ac_plot(4,it) = dim_avg(vplot_in(ind(doy.eq.id.and.zhr.eq.18))) 
  end do

  res@xyLineColors = (/"black","blue","red","darkgreen","brown"/)

  plotc = gsn_csm_xy (wks,ac_doy,runave(ac_plot,drunave,-1),res)

; Put a little text on for a key.
  
  duml = new(nlines-1,"graphic")
  xtext = 0.93*max(ac_doy)

  do il=0,nlines-2
    ytext = (0.97-il*0.042)*(res@trYMaxF-res@trYMinF)+res@trYMinF
    txres@txFontColor =  res@xyLineColors(il+1)
    duml(il) = gsn_add_text(wks,plotc,hrs(il)+"Z",xtext,ytext,txres) 
  end do

  delete(ac_plot)

end if

;
; Last thing is to printt the CAM lat/lon on bottomleft of plot
; For sanity

if (rdata@cam) then
  txres@txFontHeightF = 0.015
  xtext = 0.28*max(ac_doy)
  ytext = 0.03*(res@trYMaxF-res@trYMinF)+res@trYMinF

  txres@txFontColor =  1
  dumc = gsn_add_text(wks,plotc,"CAM grid-point: "+sprintf("%5.2f",vplot_in@lat)+"N, "+sprintf("%5.2f",(vplot_in@lon))+"E, "+sprintf("%4.0f",vplot_in@cam_elev)+"m",xtext,ytext,txres) 
end if

;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Plot observations here. ;
;;;;;;;;;;;;;;;;;;;;;;;;;;;

;dobs = (/"CPC","GPCP","TRMM","Livneh","PRISM"/)
;dyrs = (/"1981-2010","1979-2009","1998-2009","1981-2010","1981-2010"/)

dobs = (/"GPCP","TRMM"/)
dyrs = (/"1979-2009","1998-2009"/)

lcols = (/"gray35","gray35","gray25","gray25","gray25"/)
ldash =  (/1,2,14,5,3/)

nobs = dimsizes(dobs)
print("-----------------------------------------")
print("---- Plotting Precip Datasets ----")
do io=0,nobs-1
  print("** "+dobs(io)+" **")
end do


if (any(dobs.eq.(/"CPC"/))) then
  cpc_ave := get_obs_precip("CPC",rdata)
end if

if (any(dobs.eq.(/"Livneh"/))) then
  livneh_ave := get_obs_precip("Livneh",rdata)
end if

if (any(dobs.eq.(/"PRISM"/))) then
  prism_ave := get_obs_precip("PRISM",rdata)
end if

if (any(dobs.eq.(/"TRMM"/))) then ; Construct from daily aves. for now. 
  obs_prec := get_obs_precip("TRMM",rdata)
  date := toint(cd_calendar(obs_prec&time,0)) 
  doy := day_of_year(date(:,0),date(:,1),date(:,2))
  ndoys = max(doy)
  obs_ave = new((ndoys),"float")
  
  do it=0,ndoys-1
    id = it+1
    obs_ave(it) = dim_avg(obs_prec(ind(doy.eq.id))) ; mean
  end do
  trmm_ave = obs_ave
 
end if

if (any(dobs.eq.(/"GPCP"/))) then ; Construct from daily aves. for now. 
  obs_prec := get_obs_precip("GPCP",rdata)
  date := toint(cd_calendar(obs_prec&time,0)) 
  doy := day_of_year(date(:,0),date(:,1),date(:,2))
  ndoys = max(doy)
  obs_ave = new((ndoys),"float")
  
  do it=0,ndoys-1
    id = it+1
    obs_ave(it) = dim_avg(obs_prec(ind(doy.eq.id))) ; mean
  end do
  gpcp_ave = obs_ave
end if

; Plot alll obs. daily climo. separatley as some are 365 some 366 days. 


do iobs=0,nobs-1
  if (dobs(iobs).eq."CPC") then
    obs_ave := cpc_ave
  end if
  if (dobs(iobs).eq."TRMM") then
    obs_ave := trmm_ave
  end if
  if (dobs(iobs).eq."GPCP") then
    obs_ave := gpcp_ave
  end if
  if (dobs(iobs).eq."Livneh") then
    obs_ave := livneh_ave
  end if
  if (dobs(iobs).eq."PRISM") then
    obs_ave := prism_ave
  end if

  res@xyLineColors(0) = lcols(iobs)
  res@xyDashPatterns(0) = ldash(iobs)
  ploto = gsn_csm_xy (wks,ac_doy,runave(obs_ave,drunave,-1),res)
  overlay(plotc,ploto)
  delete(ploto)
end do


; Quick legend with obs. lines.

legend1 = create "Legend" legendClass wks 
"vpXF"                     : 0.18                   ; orientation on page
"vpYF"                     : 0.89
"vpWidthF"                 : 0.15                     ; width
"vpHeightF"                : 0.12                     ; height
"lgPerimOn"                : False                  ; no perimeter
"lgItemCount"              : nobs                       ; how many
"lgLabelStrings"           : " "+dobs+" ("+dyrs+")"  ; Labels
"lgLabelsOn"               : True                   ; no default lables
"lgMonoDashIndex"          : False
"lgMonoLineThickness"      : True
"lgLineThicknessF"         : 3 
"lgDashIndexes"            : ldash
"lgLabelFontHeightF"       : 0.016                   ; font height
"lgLineColors"             : lcols
"lgMonoLineLabelFontColor" : False                    ; one label color
"lgItemOrder"              : ispan(nobs-1,0,1)    ; Reverse order 
"lgAutoManage"             : False
"lgLabelJust"              : "CenterLeft"
end create

draw(legend1)

draw(plotc)


end ;station_precip_plot

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure station_tp_plot(wks,vplot_in,rdata)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

;;;;; Plot 1d raob data

;1. Grab date info for plotting  
  
date = toint(cd_calendar(vplot_in&time,0))
years = date(:,0)
year0 = rdata@year_first
year1 = rdata@year_last
drunave = rdata@drunave
std2d = rdata@lp2d_std

; Day of year for plot
doy = day_of_year(date(:,0),date(:,1),date(:,2))
ndoys = max(doy)

;Levels 
nlevs = dimsizes(vplot_in&lev)

res                     = True                   ; plot mods desired
res@cname = vplot_in@cname
res@lp2d_nomean = rdata@lp2d_nomean

; Lets grab some common resources we shouldn't repeat for the time axis.
raob_plot_res_tp(res)
delete(res@lp2d_nomean)

;Local resources.

res@tiMainString        = vplot_in@long_name
res@tiYAxisString       = "Pressure (mb)"

res@tiMainString = rdata@station+" ("+rdata@lat+" lat, "+rdata@lon+" lon, "+rdata@elev+"m elev.)"
res@gsnCenterStringFontHeightF = 0.015
res@gsnCenterString = vplot_in@long_name+" ("+vplot_in@units+")"
res@gsnDraw = False
res@gsnFrame = False

;res@trYMinF = min(vplot_in&lev)
;res@trYMaxF = max(vplot_in&lev)
res@trYMinF = 400.

; Text
txres = True
txres@txFontHeightF = 0.02  

; Day of year array (w/ leap) 
ac_doy = fspan(1,ndoys,ndoys)

; Trim for years (maskk out will _Fill)

do ilev=0,nlevs-1
  vplot_in(:,ilev) = where(years.ge.year0.and.years.le.year1,vplot_in(:,ilev),vplot_in@_FillValue)
end do



; Date hour.
 zhr = date(:,3) ; Z hour of day.

; ########################################

if (rdata@lp2d_mean) then

; Set up mean array.
  ac_plot = new((/nlevs,ndoys/),"float")
  ac_plot!0 = "lev"
  ac_plot&lev = vplot_in&lev
  ac_plot&lev@units = "mb"
  ac_plot!1 = "doy"


; To be consistant just take 00Z/12Z periods

  do it=0,ndoys-1
    id = it+1
    ac_plot(:,it) = dim_avg_n(vplot_in(ind(doy.eq.id.and.(zhr.eq.0.or.zhr.eq.12)),:),0) ; mean
  end do

; Attach +/- one std for each day.  

  if (rdata@cam) then
    res@gsnCenterString = vplot_in@long_name+" ("+vplot_in@units+") - CAM grid point: "+sprintf("%5.2f",vplot_in@lat)+"N, "+sprintf("%5.2f",(vplot_in@lon))+"E, "+sprintf("%4.0f",vplot_in@cam_elev)+"m"
  end if
  
  if (rdata@lp2d_nomean) then
    ac_plot = dim_rmvmed_n_Wrap(ac_plot,1)
  end if
  
  plot = gsn_csm_pres_hgt (wks,runave_n_Wrap(ac_plot,drunave,-1,1),res) 
  
  delete(ac_plot)
end if

;############################################

if (rdata@lp2d_std) then

; Set up mean array.
  ac_plot = new((/nlevs,ndoys/),"float")
  ac_plot!0 = "lev"
  ac_plot&lev = vplot_in&lev
  ac_plot&lev@units = "mb"
  ac_plot!1 = "doy"
  
; std
  do it=0,ndoys-1
    id = it+1
    ac_plot(:,it) = dim_stddev_n(vplot_in(ind(doy.eq.id.and.(zhr.eq.0.or.zhr.eq.12)),:),0) ; mean
  end do

; Add standard deviation contouring.

;  res@cnMinLevelValF       =  0.5                  ; set minn contour level
;  res@cnMaxLevelValF       =  5                   ; set maxx contour level
;  res@cnLevelSpacingF      =  0.25                  ; set contour spacing
  res@cnLineColor = "blue"
  res@cnLineLabelFontColor = "blue"

  plots = gsn_csm_pres_hgt (wks,runave_n_Wrap(ac_plot,drunave,-1,1),res) 

  if (rdata@cam) then
    res@gsnCenterString = vplot_in@long_name+" ("+vplot_in@units+") - CAM grid point: "+sprintf("%5.2f",vplot_in@lat)+"N, "+sprintf("%5.2f",(vplot_in@lon))+"E, "+sprintf("%4.0f",vplot_in@cam_elev)+"m"
  end if

  delete(ac_plot)
  overlay(plot,plots)

end if

;############################################

if (rdata@lp2d_0012Z) then

; Set up mean array.
  ac_plot = new((/nlevs,ndoys/),"float")
  ac_plot!0 = "lev"
  ac_plot&lev = vplot_in&lev
  ac_plot&lev@units = "mb"
  ac_plot!1 = "doy"

;Graphics
  
  nplots = 3
  plot = new(nplots,"graphic")
  pan = True
  pan@txString = rdata@station+" ("+rdata@lat+" lat, "+rdata@lon+" lon, "+rdata@elev+"m elev.) - "+vplot_in@long_name+" ("+vplot_in@units+")"

  pan@gsnMaximize = True
  pan@txFontHeightF = 0.015
  delete(res@tiMainString)

  res@gsnDraw = False
  res@gsnFrame = False
  res@gsnCenterStringFontHeightF = 0.04

  ac_plot0 = ac_plot
  ac_plot1 = ac_plot
  ac_plotd = ac_plot
  
  do it=0,ndoys-1
    id = it+1
    ac_plot0(:,it) = dim_avg_n(vplot_in(ind(doy.eq.id.and.(zhr.eq.0)),:),0) ; mean
    ac_plot1(:,it) = dim_avg_n(vplot_in(ind(doy.eq.id.and.(zhr.eq.12)),:),0) ; mean
  end do

  ac_plotd = ac_plot0-ac_plot1

  res@gsnCenterString = "00Z"
  plot(0) = gsn_csm_pres_hgt (wks,runave_n_Wrap(ac_plot0,drunave,-1,1),res)
  res@gsnCenterString = "12Z"
  plot(1) = gsn_csm_pres_hgt (wks,runave_n_Wrap(ac_plot1,drunave,-1,1),res)

  res@gsnCenterString = "00Z minus 12Z"
;  res@cnLevelSpacingF      =  0.5*res@cnLevelSpacingF

  
  if (any(vplot_in@cname.eq.(/"Q"/))) then
    res@cnLevelSelectionMode = "ExplicitLevels"
    res@cnLevels = (/-10.,-8.,-5.,-4.,-3.,-2.,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.,1.5,2.,3,4,5,8,10./)
    
  delete([/res@cnMinLevelValF,res@cnMaxLevelValF,res@cnLevelSpacingF/])          

;    res@cnMinLevelValF       =  -5                  ; set minn contour level
;    res@cnMaxLevelValF       =  5.                  ; set maxx contour level
;    res@cnLevelSpacingF      =  0.25                  ; set contour spacing
  end if

  if (any(vplot_in@cname.eq.(/"T"/))) then
    res@cnMinLevelValF       =  -10                  ; set minn contour level
    res@cnMaxLevelValF       =  10.                  ; set maxx contour level
    res@cnLevelSpacingF      =  0.5                  ; set contour spacing
  end if

  if (any(vplot_in@cname.eq.(/"VMAG","U"/))) then
    

    res@cnMinLevelValF       =  -20.                  ; set minn contour level
    res@cnMaxLevelValF       =  20.                  ; set maxx contour level
    res@cnLevelSpacingF      =  2.                  ; set contour spacing
  end if
  
  if (any(vplot_in@cname.eq.(/"V","DPD"/))) then
    res@cnLevelSelectionMode = "ExplicitLevels"
    res@cnLevels = (/-10.,-8.,-5.,-4.,-3.,-2.,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.,1.5,2.,3,4,5,8,10./)
    
    delete([/res@cnMinLevelValF,res@cnMaxLevelValF,res@cnLevelSpacingF/])     
 

;    res@cnMinLevelValF       =  -5.                  ; set minn contour level
;    res@cnMaxLevelValF       =  5.                  ; set maxx contour level
;    res@cnLevelSpacingF      =  0.5                  ; set contour spacing
  end if
  
  if (any(vplot_in@cname.eq.(/"RELHUM"/))) then
    res@cnMinLevelValF       =  -20                  ; set minn contour level
    res@cnMaxLevelValF       =  20                  ; set maxx contour level
    res@cnLevelSpacingF      =  2.                  ; set contour spacing
  end if
  

  plot(2) = gsn_csm_pres_hgt (wks,runave_n_Wrap(ac_plotd,drunave,-1,1),res)

  if (rdata@cam) then
    ctxt = True
    ctxt@txFontHeightF = 0.012  

    camtxt = "CAM grid point: "+sprintf("%5.2f",vplot_in@lat)+"N, "+sprintf("%5.2f",(vplot_in@lon))+"E, "+sprintf("%4.0f",vplot_in@cam_elev)+"m"
;    dum1 = gsn_add_text(wks,plot(0),camtxt,xtext,ytext,True) 
    gsn_text_ndc(wks,camtxt,0.5,0.31,ctxt)	

  end if
end if

if (.not.isdefined("plot")) then
  print("-No 2D plots have been selected ...EXITING")
  exit
end if

if (rdata@lp2d_0012Z) then
  gsn_panel(wks,plot,(/1,nplots/),pan)
else
  draw(plot)
end if


end ; station_tp_plot




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function raob_sig2p(var_sig,splevs,mplevs)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

begin

; Interpolate to selected "mandatory" levels from significant T/U levels on raobs file.
; Need some quality control 

; Ensure correct fill values and lev units for interpolation.
  
  var_sig@_FillValue = default_fillvalue(typeof(var_sig))   
  splevs@_FillValue = default_fillvalue(typeof(splevs))      
  splevs@missing_value = default_fillvalue(typeof(splevs))
  
  splevs = (/0.01*splevs/) ; ->mb
  mplevs = dimsizes(splevs)
  
; Need to make difficult decisions about what to doo with data.

  print("-Interpolate alll ascents from 'significant' levels to 'mandatory'-type levels...could take a while")
  do it=0,nt-1
    var_tmp(it,:) = int2p(splevs(it,:),var_sig(it,:),mplevs,-1) ; -1=log interp.
  end do
  print("-Done...")
  
  var_tmp!0 = var_out!0
  var_tmp!1 = "lev"
  var_tmp&lev = mplevs
  var_tmp&lev@units = "mb"
  copy_VarCoords_1(var_out,var_tmp) ; Copy time coord.
  delete(var_out)      
  var_out = var_tmp        
  delete(var_tmp)

  return(var_out)
  
end ; raob_sig2p


