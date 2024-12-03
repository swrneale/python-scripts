'''
    Utils used for plotting single column information from SCAM runs
'''

import numpy as np

def corrcoef_ignore_nan(x, y):
    # Mask invalid values
    mask = ~np.isnan(x) & ~np.isnan(y)
    
    # Apply mask to both arrays
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Compute the correlation coefficient
    corr_matrix = np.corrcoef(x_clean, y_clean)

    print("Correlation coefficient ignoring NaNs:", corr_matrix)
    
    return corr_matrix[0, 1]



""" ########################### """
""" ###     FUNCTIONS     ##### """
""" ########################### """
def moving_average(x, w):
    return np.convolve(x, np.ones(w),'same') / w