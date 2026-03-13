import cdsapi
from datetime import datetime, timedelta

def generate_dates(year, month):
    # Create a datetime object for the first day of the given month
    start_date = datetime(year, month, 1)
    
    # Find the next month to determine the end date
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    # Calculate the number of days in the month
    delta = end_date - start_date
    
    # Generate a list of dates in the format yyyy-mm-dd
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days)]

    # Concatenate the dates into a single string separated by '/'
    concatenated_dates = '/'.join(dates)

    # Info.
    print('-For year = '+str(year)+' and month = '+str(month))
    print(concatenated_dates)
    
    return concatenated_dates






# Script to download on model levels
do_cds = True

#var_get = 'precip';  var_name = 'total_precipitation'
var_get = 'cape';  var_name = 'convective_available_potential_energy'

year0 = 1979
year1 = 2013


#######

dir_out = '/glade/derecho/scratch/rneale/ERA5/download/dcycle_3hrave/'+var_get+'/'

years_list = list(range(year0, year1 + 1))

c = cdsapi.Client()

print()
print(' - DOWNLOADING -')

for yr in range(year0,year1+1):

        print('Year = ',yr)
        

        mfile_out =   dir_out+var_get+'_'+str(yr)+'_ytest_era5_modelevs.grib' 
        print('-Output file '+mfile_out)
        
        c.retrieve('reanalysis-era5-single-levels',{
                'product_type': 'reanalysis',
                'variable': var_name,
                'year': str(yr),
                'month': [f"{m:02d}" for m in range(1, 13)],
                'day': [f"{d:02d}" for d in range(1, 32)],
                'time': [f"{h:02d}:00" for h in range(24)],
                'grid': [1.0, 1.0],
                'format': 'netcdf',
                },
                mfile_out )
