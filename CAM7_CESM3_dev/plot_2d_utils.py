
def get_indexes(month_or_season):
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    season_mapping = {
        'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11]
    }
    
    month_or_season = month_or_season.lower()
    
    if month_or_season in month_mapping:
        return [month_mapping[month_or_season]]
    elif month_or_season in season_mapping:
        return season_mapping[month_or_season]
    else:
        return None