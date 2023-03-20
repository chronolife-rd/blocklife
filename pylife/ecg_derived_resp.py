import numpy as np
from scipy.interpolate import CubicSpline

def extract_edr(ecg, peaks, fs, is_reversed_ecg=False):
    points_min = []
    points_max = []
    offset = 30
    for k in range(1, len(peaks)):
        max_ = peaks[k] - offset + np.argmax(ecg[peaks[k]-offset:
                                                 peaks[k]+offset])
        min_ = peaks[k] - offset + np.argmin(ecg[peaks[k]-offset:
                                                 peaks[k]+offset])
        points_min.append(min_)
        points_max.append(max_)
    points_min = np.unique(points_min)
    points_max = np.unique(points_max)
    if is_reversed_ecg:
        point_extrem = points_min
    else:
        point_extrem = points_max
        
    xs = np.linspace(0, len(ecg)/fs, len(ecg))
    e_min = CubicSpline(xs[point_extrem], ecg[point_extrem])
    return e_min(xs)
