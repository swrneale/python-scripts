
import cdsapi



yr0 = 1940
yr1 = 1978

#var_3d = "u"  ; var_3d_name = "u_component_of_wind"
#var_3d = "v"  ; var_3d_name = "v_component_of_wind"
#var_3d = "omega"  ; var_3d_name = "vertical_velocity"
#var_3d = "t"  ; var_3d_name = "temperature"
#var_3d = "q" ; var_3d_name = "specific_humidity"
#var_3d = "z" ; var_3d_name = "geopotential"
#var_3d = "rh" ; var_3d_name = "relative_humidity"
var_3d = "div" ; var_3d_name = "divergence"
#var_3d = "cloud" ; var_3d_name = "fraction_of_cloud_cover"
#var_3d = "vort" ; var_3d_name = "vorticity"




dir_out = "/glade/derecho/scratch/rneale/ERA5/mmean/0.25deg/"+var_3d+"/"

#plevels = ["1000"]
plevels = [
            "1000", "925", "850", "700", "600", "500",
            "400", "300", "250", "200", "150", "100",
            "70", "50", "30", "20"
           ]

months = [ "01","02","03","04","05","06","07","08","09","10","11","12"]

c = cdsapi.Client(timeout=600,quiet=False,debug=True)

for this_yr in range(yr0,yr1+1):

    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": var_3d_name,
            "pressure_level" : plevels,
            "year": this_yr,
            "month" : months,
            "time": "00:00",
            "format": "netcdf"
        },
        dir_out+var_3d+"_test_era5_monthly_"+str(this_yr)+".nc")



