import numpy as np
from pylife.filters import savitzky_golay
from pylife.detection import detect_peaks
from pylife.useful import convert_sec_to_min_freq
from pylife.filters import smooth_rr

from pylife.env import get_env
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import matplotlib.pyplot as plt


def compute_ppm(times, sig, fs, peaks, window_time, over_nb_seconds=60):
    """ Compute number of peaks per minute (bpm or rpm)

        Parameters
        ---------------------------------
        sig: Ecg or respiratory signal
        fs: sampling frequency
        peaks: peaks positions
        window_time: Compute hr per minute every window seconds

        Returns
        ---------------------------------
        ppm_mean: mean of peaks per minutes
        ppm_var: std of peaks per minutes
        ppm_times: array of tuple representing timestamps of ppm computing

    """

    rate_mean, rate_var, rate_times,\
        rate_mean_clean, rate_var_clean = compute_rate_pm(times, sig, fs, peaks,
                                                          window_time,
                                                          over_nb_seconds=over_nb_seconds)
    rate_mean = rate_mean[np.isnan(rate_mean) == False]
    rate_var = rate_var[np.isnan(rate_var) == False]
    rate_mean_clean = rate_mean_clean[np.isnan(rate_mean_clean) == False]
    rate_var_clean = rate_var_clean[np.isnan(rate_var_clean) == False]
    ppm_mean = convert_sec_to_min_freq(rate_mean)
    rate_mean_clean = convert_sec_to_min_freq(rate_mean_clean)
    ppm_var = convert_sec_to_min_freq(np.array(rate_mean) + np.array(rate_var))
    ppm_var = ppm_mean - ppm_var
    ppm_times = rate_times

    return ppm_mean, ppm_var, ppm_times, rate_mean, rate_var, rate_mean_clean, rate_var_clean


def compute_ppm_unwrap(times, sig, fs, peaks, window_time, over_nb_seconds=60):
    """ Compute number of peaks per minute (bpm or rpm)

        Parameters
        ---------------------------------
        sig: Ecg or respiratory signal
        fs: sampling frequency
        peaks: peaks positions
        window_time: Compute hr per minute every window seconds

        Returns
        ---------------------------------
        ppm_mean: mean of peaks per minutes
        ppm_var: std of peaks per minutes
        ppm_times: array of tuple representing timestamps of ppm computing

    """
    ppm_mean_s = []
    ppm_var_s = []
    ppm_times_s = []
    rate_mean_s = []
    rate_var_s = []
    rate_mean_clean_s = []
    rate_var_clean_s = []

    for id_seg, seg in enumerate(sig):
        times_seg = times[id_seg]
        peaks_seg = peaks[id_seg]
        ppm_mean, ppm_var, ppm_times,\
            rate_mean, rate_var, rate_mean_clean, rate_var_clean = compute_ppm(times_seg, seg, fs, peaks_seg,
                                                                               window_time=window_time,
                                                                               over_nb_seconds=over_nb_seconds)
        ppm_mean_s.append(ppm_mean)
        ppm_var_s.append(ppm_var)
        ppm_times_s.append(ppm_times)
        rate_mean_s.append(rate_mean)
        rate_var_s.append(rate_var)
        rate_mean_clean_s.append(rate_mean_clean)
        rate_var_clean_s.append(rate_var_clean)

    return ppm_mean_s, ppm_var_s, ppm_times_s, rate_mean_s, rate_var_s, rate_mean_clean_s, rate_var_clean_s


def compute_rate_pm(times, sig, fs, peaks, window_time, over_nb_seconds=60):
    """ Compute rate per minute

        Parameters
        ---------------------------------
        sig: Ecg or respiratory signal
        fs: sampling frequency
        peaks: peaks positions
        window_time: Compute hr per minute every window seconds

        Returns
        ---------------------------------
        rate_mean: mean of rate
        rate_var: std of rate
        rate_times: array of tuple representing timestamps of rate computing

    """
    rate_mean = []
    rate_var = []
    rate_mean_clean = []
    rate_var_clean = []
    rate_times = []
    sample_ = np.arange(0, len(sig), window_time*fs)
    peaks = np.array(peaks)
    for i in sample_:
        if i < int(over_nb_seconds)/2*fs:
            p = peaks[peaks < over_nb_seconds/2*fs+i]
            rr_intervals = []
            for j in range(1, len(p)):
                rr_intervals.append((p[j]-p[j-1])/fs)            
            rate_mean.append(np.median(rr_intervals))
            rate_var.append(np.std(rr_intervals))
            rr_intervals_clean = smooth_rr(rr_intervals)
            rate_mean_clean.append(np.mean(rr_intervals_clean))
            rate_var_clean.append(np.std(rr_intervals_clean))
            
            if len(rr_intervals) > 0:
                imax = min([int(over_nb_seconds/2)*fs+i, len(times)-1])
                rate_times.append((times[0], 
                                   times[imax]))
        elif i > len(sig)-int(over_nb_seconds)/2*fs:
            p = peaks[peaks > i]
            rr_intervals = []
            for j in range(1, len(p)):
                rr_intervals.append((p[j]-p[j-1])/fs)
            rr_intervals_clean = smooth_rr(rr_intervals)
            rate_mean.append(np.median(rr_intervals))
            rate_var.append(np.std(rr_intervals))
            rate_mean_clean.append(np.mean(rr_intervals_clean))
            rate_var_clean.append(np.std(rr_intervals_clean))
            if len(rr_intervals) > 0:
                rate_times.append((times[i], 
                                   times[-1]))
        else:
            p = peaks[peaks > i-int(over_nb_seconds/2)*fs]

            p = p[p < i+int(over_nb_seconds)/2*fs]
            rr_intervals = []
            for j in range(1, len(p)):
                rr_intervals.append((p[j]-p[j-1])/fs)
            rate_mean.append(np.median(rr_intervals))
            rate_var.append(np.std(rr_intervals))
            rr_intervals_clean = smooth_rr(rr_intervals)
            rate_mean_clean.append(np.mean(rr_intervals_clean))
            rate_var_clean.append(np.std(rr_intervals_clean))
            if len(rr_intervals) > 0:
                rate_times.append((times[i-int(over_nb_seconds/2)*fs], 
                                   times[i+int(over_nb_seconds/2)*fs-1]))
    
    return np.array(rate_mean), np.array(rate_var), np.array(rate_times), np.array(rate_mean_clean), np.array(rate_var_clean)


def compute_rate_pm_unwrap(times, sig, fs, peaks, window_time, over_nb_seconds=60):
    """ Compute rate per minute

        Parameters
        ---------------------------------
        sig: Ecg or respiratory signal
        fs: sampling frequency
        peaks: peaks positions
        window_time: Compute hr per minute every window seconds

        Returns
        ---------------------------------
        rate: mean of rate
        rate_var: std of rate
    """
    rate_mean_s = []
    rate_var_s = []
    rate_times_s = []

    for id_seg, seg in enumerate(sig):
        times_seg = times[id_seg]
        peaks_seg = peaks[id_seg]
        rate_mean, rate_var,\
            rate_times = compute_rate_pm(times_seg, seg, fs, peaks_seg,
                                         window_time=window_time,
                                         over_nb_seconds=60)
        rate_mean_s.append(rate_mean)
        rate_var_s.append(rate_var)
        rate_times_s.append(rate_times)

    return rate_mean_s, rate_var_s, rate_times_s


def compute_heart_rate(sig, fs, peaks, window_time=None):
    """ Compute heart rate for each sample

    Parameters
    -----------------------
    sig: signal segment
    fs: sampling frequency
    peaks: peaks positions of qrs complex

    Returns
    -----------------------
    Heart rates
    heart rate variability

    """
    hr_s = []
    hrv_s = []

    if window_time is None:
        hr_s = np.mean((peaks[1:] - peaks[:-1])/fs)
        hrv_s = np.std((peaks[1:] - peaks[:-1])/fs)

    else:
        window_size = window_time*fs
        imax_w = 0.0
        for imin_w in range(0, len(sig)-window_size, window_size):
            imax_w = imin_w + window_size
            ipeaks_min = np.where(peaks >= imin_w)
            ipeaks_max = np.where(peaks <= imax_w)
            ipeaks = np.intersect1d(ipeaks_min, ipeaks_max)
            peaks_w = peaks[ipeaks]
            if len(peaks_w) > 1:
                hr = np.mean((peaks_w[1:] - peaks_w[:-1])/fs)
                hrv = np.std((peaks_w[1:] - peaks_w[:-1])/fs)
                hr_s.append(hr)
                hrv_s.append(hrv)

        if len(sig) - imax_w > fs:
            ipeaks_min = np.where(peaks >= imax_w)
            ipeaks_max = np.where(peaks < len(sig))
            ipeaks = np.intersect1d(ipeaks_min, ipeaks_max)
            peaks_w = peaks[ipeaks]
            if len(peaks_w) > 1:
                hr = np.mean((peaks_w[1:] - peaks_w[:-1])/fs)
                hrv = np.std((peaks_w[1:] - peaks_w[:-1])/fs)
                hr_s.append(hr)
                hrv_s.append(hrv)

    hr_s = np.array(hr_s)
    hrv_s = np.array(hrv_s)

    return hr_s, hrv_s


def compute_heart_rate_unwrap(sig, fs, peaks, window_time=None):
    """ Compute heart rate for each sample

    Parameters
    -----------------------
    sig: signal unwrapped
    fs: sampling frequency
    peaks: peaks positions of qrs complex

    Returns
    -----------------------
    Heart rates
    heart rate variability
    peaks detected

    """
    hr_s = []
    hrv_s = []

    for id_seg, seg in enumerate(sig):
        peaks_seg = peaks[id_seg]
        hr, hrv = compute_heart_rate(seg, fs, peaks_seg,
                                     window_time=window_time)
        hr_s.append(hr)
        hrv_s.append(hrv)

    return hr_s, hrv_s


def compute_breath_rate(sig, name_breath, activity, fs, show_plot=False):
    """ Compute respiratory frequency
    Input: time, data, fs, mpd, threshold_peaks
    time: timestamps
    sig: signal
    name_breath : breath_1 if thoracic, breath_2 if abdominal
    activity : True if subject is walking
    fs
    Output: respiration sampled at fs
    """
    if name_breath == 'breath_1':
        if activity:
            window = 2*int(fs*4.55/2)+1
            mpd_p = int(fs)
            mpd_v = int(fs*0.75)
            threshod_diff = 70
        else:
            window = 2*int(fs*2.05/2)+1
            mpd_p = int(fs*1.75)
            mpd_v = int(fs*0.75)
            threshod_diff = 70
    else:
        if activity:
            window = 2*int(fs*4.55/2)+1
            mpd_p = int(fs)
            mpd_v = int(fs*1.25)
            threshod_diff = 20
        else:
            window = 2*int(fs*2.05/2)+1
            mpd_p = int(fs*0.5)
            mpd_v = int(fs)
            threshod_diff = 75

    smooth = savitzky_golay(sig, window, 2)
    peaks = detect_peaks(smooth, mph=0, mpd=mpd_p, threshold=0, edge="rising",
                         kpsh=False, valley=False)
    valley = detect_peaks(smooth, mph=-10000000000, mpd=mpd_v, threshold=0,
                          edge="rising", kpsh=False, valley=True)
    remove_valleys_to_close_to_peaks = []
    if peaks.tolist():
        for id_valley in valley:
            closest_peaks = np.argmin(abs(id_valley-peaks))
            if abs(smooth[id_valley]-smooth[peaks[closest_peaks]])\
               < threshod_diff:
                remove_valleys_to_close_to_peaks.append(id_valley)
    new_valleys = []
    for id_valley in valley:
        if id_valley not in remove_valleys_to_close_to_peaks:
            new_valleys.append(id_valley)

    remove_false_valleys = []
    for id_valley in range(1, len(new_valleys)-1):
        current_ = new_valleys[id_valley]
        past_ = new_valleys[id_valley-1]
        future_ = new_valleys[id_valley+1]
        if smooth[current_] > smooth[past_] and\
                smooth[current_] > smooth[future_]:
            remove_false_valleys.append(new_valleys[id_valley])
    true_valleys = []
    for id_valley in new_valleys:
        if id_valley not in remove_false_valleys:
            true_valleys.append(id_valley)
    if DEV:    
        if show_plot:
            x_s = np.linspace(0, len(smooth)/fs, len(smooth))
            plt.figure()
            plt.plot(x_s, smooth)
            for id_valley in valley:
                plt.scatter(x_s[id_valley], smooth[id_valley], c='g')
            for id_valley in new_valleys:
                plt.scatter(x_s[id_valley], smooth[id_valley], c='b')
            for id_valley in true_valleys:
                plt.scatter(x_s[id_valley], smooth[id_valley], c='r')
            plt.title(len(true_valleys))

    return smooth, true_valleys


def pos_2_time_interval(fs, positions):
    """ Convert position between samples to time interval

    Parameters
    -----------------------
    fs: sampling frequency
    positions: positions of samples

    Returns
    -----------------------
    time_intervals

    """

    time_intervals = []
    if len(positions) > 1:
        positions = np.array(positions)
        time_intervals = (positions[1:] - positions[:-1])/fs

    return time_intervals

def pos_2_time_interval_unwrap(fs, positions):
    """ Convert position between samples to time interval

    Parameters
    -----------------------
    fs: sampling frequency
    positions: positions of samples

    Returns
    -----------------------
    time_intervals

    """
    time_intervals_s = []

    for pos in positions:
        time_intervals = pos_2_time_interval(fs, pos)
        time_intervals_s.append(time_intervals)

    return time_intervals_s
