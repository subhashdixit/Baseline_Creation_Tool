import numpy as np
import pandas as pd
from io import BytesIO

# HELPER FUNCTION

def calculate_baseline(data, window_size, smooth_factor, start_end_same:bool=True,vertical_shift:int=0):
    baseline = np.copy(data)  # Create a copy of the data to store the baseline curve
    for i in range(window_size, len(data) - window_size):
        window = data[i - window_size : i + window_size + 1]
        local_min = np.min(window)
        baseline[i] = local_min
    
    # Apply moving average smoothing
    smoothed_baseline = np.convolve(baseline, np.ones(smooth_factor) / smooth_factor, mode='same')
    
    # smoothed_baseline = np.copy(baseline)
    # for i in range(smooth_factor, len(baseline) - smooth_factor):
    #     smoothed_baseline[i] = np.mean(baseline[i - smooth_factor : i + smooth_factor + 1])
    
    # Apply vertical shift
    min_val = np.min(smoothed_baseline)
    max_val = np.max(smoothed_baseline)
    range_val = max_val - min_val
    scaled_shift = vertical_shift * range_val
    shifted_baseline = smoothed_baseline + scaled_shift

    # Keep the first and last points the same as original data
    if start_end_same:
        shifted_baseline[0] = data[0]
        shifted_baseline[-1] = data[-1]
    
    return shifted_baseline
    