load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "./raobs_utils.ncl"

begin

; Code to extract RAOBS dat from netcdd files 
; produce some plots.
; 23 Oct 15: Initial test routine to read in a sample raobs netcdf file
; and display info.

;station_names = (/"Albuquerque","Davenport","Denver","KeyWest","LakeCharles","Minneapolis","Norman","OklahomaCity","Omaha","SantaTeresa","Slidell","Springfield","Tallahassee","Topeka","Tucson","Valparaiso"/)



;station_names := (/"EasterIsland"/)
;station_names := (/"SanCristobal"/)
station_names := (/"GrandCayman"/)
;station_names := (/"Hilo"/)
;station_names := (/"Tarawa"/)

nstat = dimsizes(station_names)

do istat=0,nstat-1
  
  station = station_names(istat) 

;;; Edit out below and it will doo obs. otherwise model output.

  crun_name = "b.e11.B20TRC5CNBDRD.f09_g16.001"
  crun_type = "large_ens"

;  crun_name = "f.e20.FHIST.f09_f09.cesm2_1.001"
;  crun_type = "cam6_revert"

  vplot1d = "PS"
  vplot2d = "U"
  
  year_first = 1990
  year_last = 2015
  
  lev_type = "M" ; M-"mandatory", T-"significant temp", "W-significant W(wind)" 
  
; Raobs location
  dir_rdata = "/glade/work/rneale/data/raobs/"
; PS output dir 
  pdir0  = "/glade/work/rneale/ncl/raobs/"

;;;;;;;;;;;;;;;;;;;;;
; Generic station point info (station and CAM GP).
; AND the resources for alll sorts of graphics and averaging
  
  sdata = True
  sdata@station = station
  sdata@year_first = year_first
  sdata@year_last = year_last
  sdata@lev_type = lev_type 
  
;; Graphics options
  
; General
  sdata@drunave = 60 ; Running average for annual cycle (in days)
  
;;; PLot options (only 1 each True for 1D and 2D).
; 1D
  sdata@lp1d = True
  sdata@lp1d_0012Z = True ; Plot 00Z and 12Z separately
  sdata@lp1d_av_std = True ; Plot mean and +/- std.
  
;2D
  sdata@lp2d = True
  sdata@lp2d_mean = True; Plot mean.
  sdata@lp2d_0012Z = False ; Plot 00Z and 12Z separately + difference.
  sdata@lp2d_nomean = False ; Remove mean annual cycle from plots.
  sdata@lp2d_std = False; Overplot daily standard deviation
  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;; Station Obs. ;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
; Extract main information about this station.
  rfile = raob_file(station,dir_rdata) ; Get station id/filename
  
  print("** Plotting for years: "+year_first+"-"+year_last)
  print(" ")
  
  fr = addfile(rfile,"r") ; Open RAOBS file.
  
; Meta data about station.
  rtime = get_raob_times(fr)
  raob_info(station,fr,rtime,sdata)
  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;  Model info
;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  if (isdefined("crun_name")) then
    sdata@cam = True
    if (crun_type.eq."large_ens") then
      sdata@chan = "h2"
      dir_cdata = "/glade/work/rneale/large_ens/"
      cfile = cam_file(crun_name,dir_cdata,vplot1d,sdata)
    end if
    if (crun_type.eq."cam6_revert") then
      sdata@chan = "h1"
      dir_cdata = "/glade/work/rneale/cam6_revert/"
      cfile = cam_file(crun_name,dir_cdata,vplot1d,sdata)
    end if


    fs = addfile(cfile,"r") ; Open CAM file.
    ps_suf = "_"+crun_name
    print("Run name = "+crun_name)
    svar = cam_read(fs,vplot1d,sdata)
    
    print("-- CAM: 1D Processing done...")
  else
    fs = fr
    delete(fr)
    sdata@cam = False
    ps_suf = "_"+lev_type+"levs"
    
    svar = raob_read(fs,vplot1d,rtime,sdata)
    
    print("-- RAOB: 1D Processing done")
  end if
  
  
  
;;;;; Averaging types put in plot name ;;;;;;
  
  ps_suf1d = ps_suf
  ps_suf2d = ps_suf
  
;; 1d
  if (sdata@lp1d_av_std) then
    ps_suf1d = ps_suf1d+"_av_std"
  end if
  
  if (sdata@lp1d_0012Z) then
    ps_suf1d = ps_suf1d+"_0012Z"
  end if
  
;; 2d
  if (sdata@lp2d_mean) then
    ps_suf2d = ps_suf2d+"_av"
  end if
  
  if (sdata@lp2d_nomean) then
    ps_suf2d= ps_suf2d+"_nomean"
  end if
  
  if (sdata@lp2d_std) then
    ps_suf2d= ps_suf2d+"_std"
  end if

  if (sdata@lp2d_std) then
    ps_suf2d= ps_suf2d+"_0012Z_diff"
  end if
  
;;;;;; PLOTS ;;;;;;;;;;;;;
  
; Field to read in (use CAM names)
;1d plot
  
  print(" ")
  print("********************************************")
  print(" Making 1D plots ")
  print("********************************************")
  
  if (sdata@lp1d) then

    wks1 = gsn_open_wks("ps",pdir0+sdata@station+"-"+vplot1d+ps_suf1d+".ps") 

    
    if (sdata@cam.and.vplot1d.eq."PRECT") then ; Only plot when model data specified and PRECT requested.
      station_precip_plot(wks1,svar,sdata)
    end if
    if (vplot1d.ne."PRECT") then ; Only plot when model data specified and PRECT requested.
      station_1d_plot(wks1,svar,sdata)
    end if
    
  end if
  
 
  delete([/svar/]) ; Clean up 1D variables for next plots.
  
; 2d plot (time/pressure)
  
  if (isdefined("crun_name")) then ; CAM files are different for each var.
    cfile = cam_file(crun_name,dir_cdata,vplot2d,sdata)
    
    fs = addfile(cfile,"r") ; Open CAM file.
    
    svar = cam_read(fs,vplot2d,sdata)
    print("-- CAM: 2D Processing done...")
  else ; Raobs files contain alll variables.
    svar = raob_read(fs,vplot2d,rtime,sdata)
    print("-- RAOB: 2D Processing done")
  end if
  
  print(" ")
  print("********************************************")
  print(" Making 2D plots ")
  print("********************************************")
  
  if (sdata@lp2d) then
    wks2 = gsn_open_wks("ps",pdir0+sdata@station+"-"+vplot2d+ps_suf2d+".ps")
    station_tp_plot(wks2,svar,sdata)
  end if

  print(" ")
  print("********************************************")
  print(" Finished ")
  print("********************************************")
  
; Delete stuff for next station.

  delete([/rtime,svar/])

end do ;  stations

end