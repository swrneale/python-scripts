
def get_indexes(month_or_season):
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    season_mapping = {
        'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11], 'ann': [1,2,3,4,5,6,7,8,9,10,11,12]
    }
    
    month_or_season = month_or_season.lower()

    
    if month_or_season in month_mapping:
        return [month_mapping[month_or_season]]
    elif month_or_season in season_mapping:
        return season_mapping[month_or_season]
    else:
        return None



''' Grab Domain for 2D Plots (+modifications for fig layout for a specific region  '''

# Domain


def get_domain(pregion):


    nrow = None
    ncol = None
    
    match pregion:
    
        case 'LabSea':
            ''' Lab Sea '''
            lat_min = 35 ; lat_max = 70
            lon_min = 280 ; lon_max = 340
            plev_scale = 0.2
            aplev_scale = 0.2
            
            rlon_min = lon_min
            rlon_max = lon_max
    
    
        case 'IO':
            ''' Indian Ocean '''
            lat_min = -10 ; lat_max = 35
            lon_min = 50 ; lon_max = 120
            plev_scale = 1.
            aplev_scale = 1.
            
            rlon_min = lon_min
            rlon_max = lon_max
    
    
        case 'US':
            ''' USA '''
            lat_min = 25 ; lat_max = 55
            lon_min = -120 ; lon_max = -70 
            plev_scale = 0.25
            aplev_scale = 0.25
    
            rlon_min = 360+lon_min
            rlon_max = 360+lon_max
    
        case 'SAm':
            ''' South America '''
            lat_min = -40 ; lat_max = 15
            lon_min = -90 ; lon_max = -30 
            plev_scale = 0.5
            aplev_scale = 0.5
    
            rlon_min = 360.+lon_min
            rlon_max = 360.+lon_max
    
    
    
        case 'Aus':
            ''' Australia '''
            lat_min = -20 ; lat_max = 10
            lon_min = 120 ; lon_max = 150
            plev_scale = 0.5
            aplev_scale = 0.5
    
            rlon_min = lon_min
            rlon_max = lon_max
    
        case 'TP':
            lat_min = -20 ; lat_max = 20
            lon_min = 0 ; lon_max = 359.
            plev_scale = 0.5
            aplev_scale = 0.5
    
            rlon_min = lon_min
            rlon_max = lon_max
    
            nrow = 5 ; ncol = 2 # Rows and columns
    
        case 'WP':
            lat_min = -20 ; lat_max = 40
            lon_min = 110 ; lon_max = 270.
            plev_scale = 0.5
            aplev_scale = 0.5
    
            rlon_min = lon_min
            rlon_max = lon_max
    
        
            
        case 'Tropics':
            lat_min = -45 ; lat_max = 45
            lon_min = 0 ; lon_max = 360.
            plev_scale = 0.8
            aplev_scale = 1.
    
            rlon_min = lon_min
            rlon_max = lon_max
    
               
            nrow = 4 ; ncol = 2 # Rows and columns
    

   
    
    return lat_min,lat_max,lon_min,lon_max,rlon_min,rlon_max,nrow,ncol,plev_scale,aplev_scale
