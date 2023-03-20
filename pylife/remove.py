import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
from pylife.env import get_env
from pylife.detection import detect_peaks, detect_ecg_reversed
from pylife.useful import unwrap
from pylife.useful import set_list_of_list
from scipy.signal import welch
from scipy.signal.windows import hann

import pandas as pd

DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import matplotlib.pyplot as plt
    import scipy


def remove_baseline(sig, fs, inter_1=0.465, inter_2=0.945, show_plot=False):
    """ Remove baseline by substracting the result of two median filters
    to the ecg
    Input: ecg, inter_1=93, inter_2=189, show_plot
    ecg : input signal
    inter_1: window first median filter
    inter_2 : window second median filter
    Output: ecg without baseline
    """
    med = scipy.signal.medfilt(sig, 2*int(inter_1*fs/2)+1)
    med = scipy.signal.medfilt(med, 2*int(inter_2*fs/2)+1)
    if DEV:
        if show_plot:
            plt.figure()
            plt.plot(sig)
            plt.plot((sig-med))
            plt.title('baseline removal')
    return sig-med


def remove_disconnection(times, sig, fs, stat=[]):
    """ Remove disconnection of a signal and return list of signals without
    disconnection

    Parameters
    ---------------------
    time : timestamp list
    signal : signal
    fs
    stat: list of info about the signal

    Returns
    ---------------------
    new_t : list of list of timestamp
    new_seg : list of signals
    stat : list of info about the signal
    """
    new_seg = []
    new_t = []
    
    if len(times) > 0:
        
        times = np.array(times).astype('datetime64')
        sig = np.array(sig)
        diff = times[1:]-times[:-1]
        id_deco = np.where(np.array(diff) > np.timedelta64(1+int(1/fs*1E6),
                                                            'us'))[0]
        size_deco = diff[id_deco]
        lenght_signal = []
        
        if id_deco.tolist():
            new_t.append(times[0:id_deco[0]+1])
            new_seg.append(sig[0:id_deco[0]+1])
            lenght_signal.append(len(sig[0:id_deco[0]+1])/fs)
    
            for nb_deco in range(1, len(id_deco)):
                new_t.append(times[id_deco[nb_deco-1]+1:id_deco[nb_deco]+1])
                new_seg.append(sig[id_deco[nb_deco-1]+1:id_deco[nb_deco]+1])
                lenght_signal.append(len(sig[id_deco[nb_deco-1]+1:
                                             id_deco[nb_deco]+1])/fs)
    
            new_t.append(times[id_deco[-1]+1:])
            new_seg.append(sig[id_deco[-1]+1:])
            lenght_signal.append(len(sig[id_deco[-1]+1:])/fs)
            max_l = np.max(lenght_signal)
            stat.append(float(max_l))
            if len(size_deco) > 1:
                stat.append(np.max(size_deco).item().total_seconds())
                stat.append(np.mean(size_deco).item().total_seconds())
            else:
                stat.append(size_deco[0]/np.timedelta64(1, 's'))
                stat.append(size_deco[0]/np.timedelta64(1, 's'))
        else:
            stat.append(len(sig))
            stat.append('NA')
            stat.append('NA')
            new_t.append(times)
            new_seg.append(sig)
    
    new_times   = np.array(new_t)
    new_sig     = np.array(new_seg) 
    return new_times, new_sig, stat


def remove_disconnection_loss(times, sig, fs):
    """ Remove disconnection of a signal and return list of signals without
    disconnection

    Parameters
    ---------------------
    time : timestamp list
    signal : signal
    fs
    stat: list of info about the signal

    Returns
    ---------------------
    new_t : list of list of timestamp
    new_seg : list of signals
    stat : list of info about the signal
    """
    times = np.array(times).astype('datetime64')
    sig = np.array(sig)
    diff = times[1:]-times[:-1]
    id_deco = np.where(np.array(diff) != np.timedelta64(int(1/fs*1E6),
                                                        'us'))[0]
    size_deco = diff[id_deco]
    lenght_signal = []
    new_seg = []
    new_t = []
    if id_deco.tolist():
        new_t.append(times[0:id_deco[0]+1])
        new_seg.append(sig[0:id_deco[0]+1])
        lenght_signal.append(len(sig[0:id_deco[0]+1])/fs)

        for nb_deco in range(1, len(id_deco)):
            new_t.append(times[id_deco[nb_deco-1]+1:id_deco[nb_deco]+1])
            new_seg.append(sig[id_deco[nb_deco-1]+1:id_deco[nb_deco]+1])
            lenght_signal.append(len(sig[id_deco[nb_deco-1]+1:
                                         id_deco[nb_deco]+1])/fs)

        new_t.append(times[id_deco[-1]+1:])
        new_seg.append(sig[id_deco[-1]+1:])
        lenght_signal.append(len(sig[id_deco[-1]+1:])/fs)
    else:
        new_t.append(times)
        new_seg.append(sig)

    return new_t, new_seg, size_deco


def remove_disconnection_multi(times, sig0, sig1, fs, stat=[]):
    """ Remove disconnection of a signal and return list of signals without
    disconnection

    Parameters
    ---------------------
    time : timestamp list
    signal : signal
    fs
    stat: list of info about the signal

    Returns
    ---------------------
    new_t : list of list of timestamp
    new_seg : list of signals
    stat : list of info about the signal
    """
    times = np.array(times)
    sig0 = np.array(sig0)
    sig1 = np.array(sig1)
    diff = times[1:]-times[:-1]
    id_deco = np.where(np.array(diff) != np.timedelta64(int(1/fs*10E5),
                                                        'us'))[0]
    size_deco = diff[id_deco]
    lenght_signal = []
    new_seg0 = []
    new_seg1 = []
    new_t = []
    
    if id_deco.tolist():
        new_t.append(times[0:id_deco[0]])
        new_seg0.append(sig0[0:id_deco[0]])
        new_seg1.append(sig1[0:id_deco[0]])
        lenght_signal.append(len(sig0[0:id_deco[0]])/fs)

        for nb_deco in range(1, len(id_deco)):
            new_t.append(times[id_deco[nb_deco-1]+1:id_deco[nb_deco]])
            new_seg0.append(sig0[id_deco[nb_deco-1]+1:id_deco[nb_deco]])
            new_seg1.append(sig1[id_deco[nb_deco-1]+1:id_deco[nb_deco]])
            lenght_signal.append(len(sig0[id_deco[nb_deco-1]+1:
                                          id_deco[nb_deco]])/fs)

        new_t.append(times[id_deco[-1]+1:])
        new_seg0.append(sig0[id_deco[-1]+1:])
        new_seg1.append(sig1[id_deco[-1]+1:])
        lenght_signal.append(len(sig0[id_deco[-1]+1:])/fs)
        max_l = np.max(lenght_signal)
        stat.append(float(max_l))
        stat.append(np.max(size_deco).item().total_seconds())
        stat.append(np.mean(size_deco).item().total_seconds())
    else:
        stat.append(len(sig0))
        stat.append('NA')
        stat.append('NA')
        new_t.append(times)
        new_seg0.append(sig0)
        new_seg1.append(sig1)
   
    return new_t, new_seg0, new_seg1, stat

def remove_noise_unwrap(times, sig, fs=200, threshold=90000, sec=0.1,
                        min_clean_window=2, threshold_std=1000, threshold_peak=4e3,
                        threshold_std2=None):
    ''' Remove saturation using the derivative of the signal and flat
    parts of signal

    Parameters
    ---------------------------
    time_ : timestamp
    signal_ :array signal
    fs: sampling frequency
    threshold: remove signal where abs(derivative) > threshold
    sec: cut sample around the threshold
    min_clean_window: minimum time of clean signal per window (seconds)
    threshold_std: remove signal where std(derivative) > threshold_std

    Returns
    ---------------------------
    Signal without noise

    '''

    sig_clean = []
    times_clean = []
    indicators = []
    for id_seg, seg in enumerate(sig):

        times_seg = times[id_seg]
        times_seg_clean,\
            seg_clean, indic = remove_noise(times_seg, seg, fs=fs,
                                            threshold=threshold, sec=sec,
                                            min_clean_window=min_clean_window,
                                            threshold_std=threshold_std,
                                            threshold_peak=threshold_peak,
                                            threshold_std2=threshold_std2)
       
        indicators.append(indic)
        sig_clean.extend(seg_clean)
        times_clean.extend(times_seg_clean)

    return times_clean, sig_clean, indicators


def remove_noise(times, sig, fs=200, threshold=90000, sec=0.1, 
                 min_clean_window=2, threshold_std=1000, threshold_peak=4e3,
                 threshold_std2=None):
    ''' Remove saturation using the derivative of the signal and
    flat parts of signal

    Parameters
    ---------------------------
    time_ : timestamp
    signal_ :array signal
    fs: sampling frequency
    threshold: remove signal where abs(derivative) > threshold
    sec: cut sample around the threshold
    min_clean_window: minimum time of clean signal per window (seconds)
    threshold_std: remove signal where std(derivative) > threshold_std

    Returns
    ---------------------------
    Signal without noise

    '''
  
    x_s = np.linspace(0, len(sig)/fs, len(sig))
    window = round(fs*sec)
    sig_deriv = np.array((sig[1:]-sig[0:-1])/((x_s[1:]-x_s[0:-1])))
    std_ = []
    for i in range(len(sig_deriv)-fs):
        if i < fs:
            std_.append(np.std(sig_deriv[i:i+2*fs]))
        elif i > len(sig_deriv)-fs:
            std_.append(np.std(sig_deriv[i-2*fs:]))

        else:
            std_.append(np.std(sig_deriv[i-fs:i+fs]))
    for i in range(len(sig_deriv)-fs, len(sig_deriv)):
        std_.append(np.std(sig_deriv[i-fs:]))
    times = np.array(times)
    idx_nn_sat = np.where(sig_deriv < threshold)[0]
    idx_nn_flat = np.where(np.array(std_) > threshold_std)[0]
#    
#    plt.figure()
#    plt.plot(sig)
#    plt.figure()
#    plt.plot(sig_deriv)
#    plt.title('a')
#    plt.figure()
#    plt.plot(std_)
#    plt.title('b')
    
##    plt.plot(threshold*np.ones((len(sig_deriv), )))
##   
    idx_clean = np.intersect1d(idx_nn_sat, idx_nn_flat)
    if fs==200:
        idx_nn_sat2 = np.where(np.array(std_) < threshold_std2)[0]
        idx_clean = np.intersect1d(idx_clean, idx_nn_sat2)
    
    indicators = np.zeros((len(sig),))
    data = []
    new_time = []
    sep = np.where((idx_clean[1:]-idx_clean[:-1]) != 1)[0]

    window_clean_sig = min_clean_window*fs
#    print(len(idx_clean))
    flag_noisy_peak = False
    if sep.tolist():
        if len(idx_clean[0:max(sep[0]-window, 0)]) > window_clean_sig:
            ids = idx_clean[0:max(sep[0]-window, 0)]
            flag_noisy_peak = is_noisy_peak(sig_deriv[ids], fs, threshold_peak=threshold_peak)
            if not flag_noisy_peak:
                data.append(sig[ids])
                new_time.append(times[ids])
                indicators[ids] = 1
        for j in range(1, len(sep)):
            if len(sig[idx_clean[sep[j-1]+1]+window:
                       max(idx_clean[sep[j]]-window, 0)]) > window_clean_sig:
                imin = round((int((idx_clean[sep[j-1]+1]+window)/fs)+1)*fs)
                imax = round(max(idx_clean[sep[j]]-window, 0))
                flag_noisy_peak = is_noisy_peak(sig_deriv[imin:imax], fs, threshold_peak=threshold_peak)
                if not flag_noisy_peak:
                    data.append(sig[imin:imax])
                    new_time.append(times[imin:imax])
                    indicators[imin:imax] = 1
        if len(sig[idx_clean[sep[-1]+1]+window:idx_clean[-1]]) > window_clean_sig:
         
            imin = round((int((idx_clean[sep[-1]+1]+window)/fs)+1)*fs)
            flag_noisy_peak = is_noisy_peak(sig_deriv[imin:], fs, threshold_peak=threshold_peak)
           
            if not flag_noisy_peak:
                data.append(sig[imin:idx_clean[-1]])
                new_time.append(times[imin:idx_clean[-1]])
                indicators[imin:idx_clean[-1]] = 1
    else:
        if len(sig[idx_clean]) > window_clean_sig:
           
            imin = round((int(idx_clean[0]/fs)+1)*fs)
            idx_clean = idx_clean[imin:]
            flag_noisy_peak = is_noisy_peak(sig_deriv[idx_clean], fs, threshold_peak=threshold_peak)
            if not flag_noisy_peak:
                new_time.append(times[idx_clean])
                data.append(sig[idx_clean])
                indicators[idx_clean] = 1

    times_clean = []
    sig_clean = []
    for id_seg, seg in enumerate(data):

        if len(seg) != len(new_time[id_seg]):
            raise NameError('Error remove_noise: new sig and times have not the same size')

        if seg.tolist():
            sig_clean.append(seg)
            times_clean.append(new_time[id_seg])

    return times_clean, sig_clean, indicators

def is_noisy_peak(sig, fs, threshold_peak=4e3):
    """In remove_noise function. Detect if peak is noise"""
    flag_noisy_peak = False
    peaks = detect_peaks(sig, mph=None, mpd=int(0.55*fs))
    if peaks.tolist():
      peak = np.percentile(sig[peaks], 75)
      if peak < threshold_peak:
          flag_noisy_peak = True  
    
    return flag_noisy_peak

def remove_noisy_temp_unwrap(times, sig, fs, coeff_, temp_min,
                             temp_max, lcie):

    sig_clean = []
    times_clean = []
    nb_outliers_ = 0
    l_temp = 0
    indicators = []

    for id_seg, seg in enumerate(sig):
        times_seg = times[id_seg]
        times_seg_clean, seg_clean, indic = remove_noisy_temp(times_seg,
                                                              seg,
                                                              coeff_=coeff_,
                                                              temp_min=temp_min,
                                                              temp_max=temp_max,
                                                              lcie=lcie)

        nb_outliers_ += len(seg)-len(seg_clean)
        l_temp += len(seg)

        indicators.append(indic)
        if len(times_seg_clean) > 0:
            sig_clean.extend(seg_clean)
            times_clean.extend(times_seg_clean)

    if sig_clean:
        times_clean, sig_clean, _ = remove_disconnection(times_clean,
                                                         sig_clean,
                                                         fs)
    else:
        times_clean = []
        sig_clean = []

    return times_clean, sig_clean, indicators


def remove_noisy_temp(times, sig, coeff_, temp_min, temp_max, lcie):
    """Remove outliers of one temperature signal
    Input: data, coeff
    time : timestamp
    sig : signal
    coeff : remove points > mean(data)+coeff_*std(data)
    or <mean(data)-coeff_*std(data)
    Output: new_data
    new_data : temperature without outliers
    """

    indicators = np.zeros((len(sig),))
    indicators[sig > temp_max] = -1
    indicators[sig < temp_min] = -1

    new_time = []
    new_data = []
    
    if len(sig) > 0:
        mean_ = np.mean(sig[indicators != -1])
        cut_off = coeff_*np.std(sig[indicators != -1])
        lower_limit = mean_ - cut_off
        upper_limit = mean_ + cut_off
        new_data = []
        new_time = []
        new_indicators = indicators
        for i, value in enumerate(sig):
            if value >= lower_limit and value <= upper_limit and indicators[i]!=-1:
                new_data.append(value)
                new_time.append(times[i])
                new_indicators[i] = 1
    new_indicators[new_indicators < 0] = 0
    indicators = new_indicators
    if not lcie:
         new_indicators = [indicators[0]]
         for i in range(1, len(indicators)-1):
             if indicators[i] != indicators[i-1] and indicators[i] != indicators[i+1]:
                 new_indicators.append(indicators[i-1])
             else:
               new_indicators.append(indicators[i])  
         new_indicators.append(indicators[-1])
         new_indicators = np.array(new_indicators)
    else:
         new_indicators = np.array(indicators)
        
    return new_time, new_data, new_indicators


def remove_timestamps_unwrap(times, values, from_time, to_time):
    """ Remove timestamps from times and signal

    Parameters
    -----------------
    times: List of timestamps
    values: List of values
    from_time: Minimum time limit
    to_time: Maximum time limit

    Returns
    -----------------
    New times, new signals
    """

    if len(times) < 1:
        new_times = []
        new_values = []
        id_sig_kept = []
        return new_times, new_values, id_sig_kept

    # -----------------
    # 1st step: Check in all signals if tmin and tmax are coherents set
    # flag_remove_times to remove timestamp

    # Init ids of signal segments where tmin and tmax are located
    # id_seg_imin = 0
    # id_seg_imax = len(values)
    times  = set_list_of_list(times)
    values = set_list_of_list(values)
    unwrap_times = unwrap(times)
    # for t in unwrap(times):
        # unwrap_times.extend(t)

    # Find index of tmin and tmax in the unwrapped timestamps
    imin = np.argwhere(unwrap_times >= from_time)
    if len(imin) > 0:
        imin = imin[0, 0]
        # print('imin', imin)
    else:
        imin = 0

    imax = np.argwhere(unwrap_times >= to_time)
    if len(imax) > 0:
        imax = imax[0, 0]
        # print('imax', imax)
    else:
        imax = len(times[-1])

    # Set flag_remove_times to True if tmin and tmax are coherents
    flag_remove_times = False
    if imin > 0 or imax < len(unwrap_times):
        flag_remove_times = True
    else:
        new_times = times
        new_values = values
        id_sig_kept = range(len(values))

    # -----------------
    # Second step: Find tmin and tmax in the list of timestamps
    if flag_remove_times:

        # Init
        new_times = []
        new_values = []
        id_sig_kept = []
        flag_fill = False  # Flag for fill new arrays in

        for i, t in enumerate(times):
            if len(t) == 0:
                # new_times.append([])
                # new_values.append([])
                continue

            flag_tmin = False
            flag_tmax = False

            if from_time <= t[-1] and not flag_fill:
                imin = np.argwhere(t >= from_time)[0, 0]
                flag_fill = True
                flag_tmin = True
                # id_seg_imin = i

            if to_time <= t[-1]:
                imax = np.argwhere(t >= to_time)[0, 0]
                # id_seg_imax = i
                flag_tmax = True

            # if flag_fill is True, new arrays can be filled in according
            # to 4 conditions
            if flag_fill:
                id_sig_kept.append(i)
                if flag_tmin and flag_tmax and len(t[imin:imax]) > 0:
                    new_times.append(t[imin:imax])
                    new_values.append(values[i][imin:imax])
                elif not flag_tmin and not flag_tmax:
                    new_times.append(t)
                    new_values.append(values[i])
                elif flag_tmin and not flag_tmax and len(t[imin:]) > 0:
                    new_times.append(t[imin:])
                    new_values.append(values[i][imin:])
                    
                elif not flag_tmin and flag_tmax and len(t[:imax]) > 0:
                    new_times.append(t[:imax])
                    new_values.append(values[i][:imax])

                # If tmaw has been reached, break the loop
                if flag_tmax:
                    break

    # print('imin final', id_seg_imin, imin)
    # print('imax final', id_seg_imax, imax)

    return new_times, new_values, id_sig_kept


def remove_timestamps(times, values, from_time, to_time):
    """ Remove timestamps from times and signal

    Parameters
    -----------------
    times: Timestamps
    values: Values
    tmin: Minimum time limit
    tmax: Maximum time limit

    Returns
    -----------------
    New times, new signals
    """

    new_times = []
    new_values = []

    if len(times) == 0:
        return new_times, new_values

    times = np.array(times)
    values = np.array(values)

    imin = np.argwhere(times >= from_time)
    if len(imin) > 0:
        imin = imin[0, 0]
    else:
        imin = 0

    imax = np.argwhere(times >= to_time)
    if len(imax) > 0:
        imax = imax[0, 0]
    else:
        imax = -1
    new_times = times[imin:imax]
    new_values = values[imin:imax]

    return new_times, new_values


def remove_noise_with_EMD(times, sig, fs=200, window=5, add_condition=False):
    
    N = min([len(sig), fs*5*60])
    if detect_ecg_reversed(sig, fs, N=N):
        sig = -sig
    indicators_all  = []
    data_clean      = []
    times_clean     = []
    window          = window * fs
    indicators_     = np.zeros(len(sig))
    for j in range(int(len(sig) / window) + 1):
        imin    = j * window
        imax    = (j + 1) * window
        d       = sig[imin:imax]
        if len(d) == window:
            max_iters       = 0
            nb_extremas     = 1000
            zero_crossings  = 0
            mean_           = 10
            while (abs(nb_extremas - zero_crossings) > 1 and mean_ > 0.1) and\
                    max_iters < 5000:
                xs = np.linspace(0, len(d)/fs, len(d))
                maxs = detect_peaks(d, mph=None, mpd=0, threshold=0,
                                    edge='rising', kpsh=False, valley=False)
                mins = detect_peaks(d, mph=None, mpd=0, threshold=0,
                                    edge='rising', kpsh=False, valley=True)
                nb_extremas = len(maxs) + len(mins)
                if len(maxs) > 2 and len(mins) > 2:
                    zero_crossings = len(np.where(d[1:] * d[:-1] < 0)[0])
                    e_min = CubicSpline(xs[mins], d[mins])
                    e_max = CubicSpline(xs[maxs], d[maxs])
                    m = (e_min(xs)+e_max(xs))/2
                    mean_ = abs(np.mean(m))
                    d = d - m
                    max_iters += 1
                else:
                    max_iters += 5000
               
            d_square = d*d
            
            d_square_sat = d_square[fs:-fs]

            # d_square_sat[d_square_sat > 15000] = 10000
            
            d_norm = (d_square_sat - np.min(d_square_sat))\
                / (np.max(d_square_sat) - np.min(d_square_sat))
                
            # seuil = []
            # for k in range(len(d_norm)):
            #     seuil.append(np.sum(d_norm[k-50:k+50]))
            # th_ = np.where(np.array(seuil) > 50)[0]
            # sep = np.where(np.array(th_[1:] - th_[:-1]) != 1)[0]
            # seg = np.split(th_, sep)
            # s_ = []
            # for s in seg:
            #     s_.append(len(s))
            # s_1 = np.max(s_)
            s_1 = 0
           
            d_norm = d_norm[np.where(d_norm < 0.18)[0]]
            if add_condition:
                if (stats.entropy(d_norm.tolist()) > 5):
                    indicators_all.extend(np.zeros((len(sig[imin:imax]),)))
                elif (np.var(d_norm) > 0.0010):
                    indicators_all.extend(np.zeros((len(sig[imin:imax]),)))  
            if (np.var(d_norm) > 0.0010) &\
                (np.mean(d_norm) > 0.02) &\
                    (stats.entropy(d_norm.tolist()) > 5.5):
                pass    
                indicators_all.extend(np.zeros((len(sig[imin:imax]),)))
        
            elif (np.var(d_norm) > 0.002) & (np.mean(d_norm) > 0.03):
                pass
                indicators_all.extend(np.zeros((len(sig[imin:imax]),)))
                  
            elif (s_1 > 200) & (np.var(d_norm) > 0.001) &\
                 (np.mean(d_norm) > 0.02) &\
                    (stats.entropy(d_norm.tolist()) > 5.0):
                pass
                indicators_all.extend(np.zeros((len(sig[imin:imax]),)))
                 
            elif s_1 > 400:
                pass
                indicators_all.extend(np.zeros((len(sig[imin:imax]),)))
                      
            else:
                data_clean.extend(sig[imin:imax])
                times_clean.extend(times[imin:imax])
                indicators_[imin:imax] = 1
                indicators_all.extend(np.ones((len(sig[imin:imax]),)))
               
        else:
            pass
            indicators_all.extend(np.zeros((len(sig[imin:imax]),)))
            
    if len(sig) - imin < window:
        
        imax = len(sig)
        last_indicator = indicators_[len(data_clean)-1]
        if last_indicator == 1:
            data_clean.extend(sig[imin:imax])
            times_clean.extend(times[imin:imax])
            indicators_[imin:imax] = 1
            indicators_all.extend(np.ones((len(sig[imin:imax]),)))
        
    return times_clean, data_clean, indicators_, indicators_all

def remove_noise_with_emd_unwrap(times, sig, fs=200, window=5, add_condition=False):

    sig_clean = []
    times_clean = []
    indicators_ = []
    new_sig_clean = []
    new_times_clean = []
    new_indicators = []
    
    for id_seg, seg in enumerate(sig):
        times_seg = times[id_seg]
        
        times_seg_clean, seg_clean, ind_, ind_all = remove_noise_with_EMD(times_seg,
                                                                              seg,
                                                                              fs=fs,
                                                                              window=window,
                                                                              add_condition=add_condition)
        if len(times_seg_clean) > 0:
            sig_clean.append(seg_clean)
            times_clean.append(times_seg_clean)
        indicators_.append(ind_)
    
    times_clean = np.array(times_clean)
    sig_clean   = np.array(sig_clean)
    indicators  = indicators_
            
    # Remove lonely segments
    if len(times_clean) > 0:
        times_ = np.array(unwrap(times))
        indicators_emd_ = np.array(unwrap(indicators_))
        if sum(indicators_emd_) != len(times_):
            for i, times_emd in enumerate(times_clean):    
                duration = (times_emd[-1]-times_emd[0])/np.timedelta64(1,'s')
                if i > 0 and i < len(times_clean)-1:
                    times_before = times_clean[i-1]
                    times_after = times_clean[i+1]        
                    time_before = (times_emd[0]-times_before[-1])/np.timedelta64(1,'s')
                    time_after = (times_after[0]-times_emd[-1])/np.timedelta64(1,'s')
                    
                    if duration < 5 and time_before > 10 and time_after > 10:
                        start = times_emd[0]
                        stop = times_emd[-1]
                        istart = np.where(times_ >= start)[0][0]
                        istop = np.where(times_ > stop)[0][0]
                        indicators_emd_[istart : istop] = 0
              
            # Wrap clean segments for times and sig
            times_emd_ = np.array(unwrap(times_clean))
            sig_emd_ = np.array(unwrap(sig_clean))        
            times_diff = (times_emd_[1:] - times_emd_[:-1])
            time_diff_ref = np.timedelta64(int(1/fs*1e6), 'us')
            idis = np.argwhere(times_diff > time_diff_ref)
            imin = 0
            if len(idis) > 0:
                idis = idis[:,0]
                for i in idis:
                    imax = int(i) + 1
                    new_times_clean.append(times_emd_[imin:imax])
                    new_sig_clean.append(sig_emd_[imin:imax])
                    imin = imax 
                if imin < len(sig_emd_):
                    new_times_clean.append(times_emd_[imin:])
                    new_sig_clean.append(sig_emd_[imin:])
            else:
                new_times_clean = times_emd_
                new_sig_clean = sig_emd_
                
            # Wrap indicators
            times_diff = (times_[1:] - times_[:-1])
            time_diff_ref = np.timedelta64(int(1/fs*1e6), 'us')
            idis = np.argwhere(times_diff > time_diff_ref)
            imin = 0
            if len(idis) > 0:
                idis = idis[:,0]
                for i in idis:
                    imax = int(i) + 1
                    new_indicators.append(indicators_emd_[imin:imax])
                    imin = imax 
                if imin < len(times_):
                    new_indicators.append(indicators_emd_[imin:])
            else:
                new_indicators = indicators_emd_
            
            times_clean = np.array(new_times_clean, dtype=object)
            sig_clean   = np.array(new_sig_clean, dtype=object)
            indicators  = new_indicators
    return times_clean, sig_clean, indicators

def remove_peaks(peaks, indicators, fs):

    clean_peaks = []
    idx = np.where(np.array(indicators) == 1)[0]
    sep = np.where((idx[1:]-idx[:-1]) != 1)[0]
    if np.sum(indicators) != 0:
        if sep.tolist():
            asso_peaks = peaks[peaks > idx[0]]
            asso_peaks = asso_peaks[asso_peaks < idx[max(sep[0], 0)]]
            asso_peaks = np.array(asso_peaks) - idx[0]
            clean_peaks.append(asso_peaks)
            for j in range(1, len(sep)):
                id0 = idx[sep[j-1]+1]
                idmax = max(idx[sep[j]], 0)
                asso_peaks = peaks[peaks > id0]
                asso_peaks = asso_peaks[asso_peaks < idmax]
                asso_peaks = np.array(asso_peaks) - id0
                clean_peaks.append(asso_peaks)
            id0 = idx[sep[-1]+1]
            asso_peaks = peaks[peaks > id0]
            asso_peaks = asso_peaks[asso_peaks < idx[-1]]
            asso_peaks = np.array(asso_peaks) - id0
            clean_peaks.append(asso_peaks)
        else:
            id0 = idx[0]
            idmax = idx[-1]
            asso_peaks = peaks[peaks > id0]
            asso_peaks = asso_peaks[asso_peaks < idmax]
            asso_peaks = np.array(asso_peaks) - id0
            clean_peaks.append(asso_peaks)
    else:
        clean_peaks.append([])
    rr = []
    for r in range(len(clean_peaks)):
        seq = clean_peaks[r]
        for points in range(1, len(seq)):
            rr.append((seq[points]-seq[points-1])/fs)
    return rr, clean_peaks

def remove_peaks_unwrap(peaks, indicators, fs):

    rr = []
    clean_peaks = []
    for id_seg, seg in enumerate(peaks):     
        if len(peaks[id_seg]) > 3:
            rr_id_seg, clean_peaks = remove_peaks(peaks[id_seg],
                                                   indicators[id_seg], fs)
             
            rr.append(rr_id_seg)
    return unwrap(rr), clean_peaks


def remove_noise_with_psd(times, sig, fs, window=20):
    ''' Remove noise using Power Spectral Density (PSD)'''
    rpm_min = 6
    rpm_max = 40
    
    times_psd = []
    sig_psd = []
    periodograms = []
    peaks_psd = []
    indicators = []
    
    if len(sig) >= fs*window:
        for i in np.arange(0, len(sig), fs*window):
            segment = sig[i:i+fs*window]
            times_seg = times[i:i+fs*window]
            # Power Spectral density (PSD)
            N = len(times_seg)
            
            if N < fs*window:
                ind_ = np.zeros(N)
                indicators.extend(ind_)
                continue
            
            nfft = 4096
            overlap = 128
            win = hann(len(segment), True)            
            xf, periodogram = welch(segment, fs, window=win, noverlap=overlap, nfft=nfft, return_onesided=True)
            peaks = detect_peaks(periodogram, mpd=1)
            
            is_clean                    = False
            is_rpm1_or_rpm2_ok          = False
            is_peaks_ratio_1_2_ok       = False
            is_peaks_ratio_2_3_ok       = False
            is_rpm_ratio_ok             = False
            is_rpm_selected_ok          = False
            is_psd_max_selected_1_ok    = False
            
            # Select PSD and Frequency values at PSD peaks level
            psd_at_peaks    = periodogram[peaks]
            xf_at_peaks     = xf[peaks]
            
            # Selection of 2 higher PSD peaks 
            psd_at_peaks_sort   = np.sort(psd_at_peaks)
            psd_max1            = psd_at_peaks_sort[-1]
            imax1               = np.argwhere(psd_at_peaks == psd_max1)[0][0] 
            rpm1                = 60/(1/xf_at_peaks[imax1])
            
            psd_max2            = psd_at_peaks_sort[-2]
            imax2               = np.where(psd_at_peaks == psd_max2)[0][0] 
            rpm2                = 60/(1/xf_at_peaks[imax2])
            
            psd_max3            = psd_at_peaks_sort[-3]
            imax3               = np.where(psd_at_peaks == psd_max3)[0][0] 
            rpm3                = 60/(1/xf_at_peaks[imax3])
            
            if rpm1 > rpm_min:
                rpm_selected_1      = rpm1
                psd_max_selected_1  = psd_max1
                
                rpm_ratio_thr       = 2
                rpm_ratio_tol       = rpm1*(1/100)
                # if  rpm_ratio_thr - rpm_ratio_tol < rpm3/rpm1 < rpm_ratio_thr + rpm_ratio_tol:
                #     rpm_selected_2      = rpm3
                #     psd_max_selected_2  = psd_max3
                # else:
                rpm_selected_2      = rpm2
                psd_max_selected_2  = psd_max2
                
                rpm_selected_3      = rpm3
                psd_max_selected_3  = psd_max3
                is_rpm1_or_rpm2_ok  = True
                
            elif rpm1 < rpm_min and rpm2 > rpm_min:
                rpm_selected_1      = rpm2
                psd_max_selected_1  = psd_max2
                
                rpm_selected_2      = rpm3
                psd_max_selected_2  = psd_max3
                
                rpm_selected_3      = rpm1
                psd_max_selected_3  = psd_max1
                is_rpm1_or_rpm2_ok  = True
                
            if is_rpm1_or_rpm2_ok:
                psd_max_selected_thr        = 500
                psd_max_selected_master     = 1e5
                if psd_max_selected_1 > psd_max_selected_thr:
                    is_psd_max_selected_1_ok = True
                
                rpm_ratio               = rpm_selected_2/rpm_selected_1
                peaks_ratio_1_2         = psd_max_selected_2/psd_max_selected_1
                peaks_ratio_2_3         = psd_max_selected_3/psd_max_selected_2
                peaks_ratio_1_2_thr     = 0.2
                peaks_ratio_2_3_thr     = 0.75
                peaks_ratio_1_2_master  = .16
                
                if psd_max_selected_1 >= psd_max_selected_master:
                    is_rpm_ratio_ok     = True
                    rpm_ratio_thr       = -1
                    
                if peaks_ratio_1_2 <= peaks_ratio_1_2_master:
                    is_rpm_ratio_ok     = True
                    rpm_ratio_thr       = -1
                else:
                    if rpm_selected_2 > rpm_selected_1:
                        rpm_ratio_thr       = 2
                        rpm_ratio_tol       = rpm_selected_1*(1/100)
                    else:
                        rpm_ratio_thr       = .5
                        rpm_ratio_tol       = .05
                        
                    if  rpm_ratio_thr - rpm_ratio_tol < rpm_ratio < rpm_ratio_thr + rpm_ratio_tol:
                        is_rpm_ratio_ok = True
                        if rpm_selected_2 > rpm_selected_1:
                            peaks_ratio_1_2_thr = 1 - 1e3/psd_max_selected_1
                            if peaks_ratio_1_2_thr > .5:
                                peaks_ratio_1_2_thr = .5
                            
                if rpm_selected_2 > rpm_selected_1 or peaks_ratio_1_2 <= peaks_ratio_1_2_master:
                    is_peaks_ratio_1_2_ok   = peaks_ratio_1_2 <= peaks_ratio_1_2_thr
                else:
                    peaks_ratio_1_2_thr = .75
                    is_peaks_ratio_1_2_ok   = peaks_ratio_1_2 >= peaks_ratio_1_2_thr
                    
                if peaks_ratio_1_2 <= peaks_ratio_1_2_master:
                    peaks_ratio_2_3_thr     = 1
                    is_rpm_selected_ok  = rpm_min <= rpm_selected_1 <= rpm_max\
                        and rpm_min <= rpm_selected_2
                else:
                    is_rpm_selected_ok  = rpm_min <= rpm_selected_1 <= rpm_max\
                        and rpm_min <= rpm_selected_2 <= rpm_max
        
                if peaks_ratio_2_3 <= peaks_ratio_2_3_thr:
                    is_peaks_ratio_2_3_ok = True
                    
            if is_psd_max_selected_1_ok and is_rpm1_or_rpm2_ok\
                and is_peaks_ratio_1_2_ok and is_peaks_ratio_2_3_ok\
                    and is_rpm_ratio_ok and is_rpm_selected_ok:
                        is_clean    = True
                        
            if is_clean:
                ind_ = np.ones(N)
                sig_psd.extend(segment)
                times_psd.extend(times_seg)
            else:
                ind_ = np.zeros(N)  
            indicators.extend(ind_)
            periodograms.append(periodogram)
            peaks_psd.append(peaks)
            
    else:
        ind_ = np.zeros(len(sig))
        indicators.extend(ind_)

    return times_psd, sig_psd, indicators

def remove_noise_with_psd_unwrap(times, sig, fs, window=20):
    
    times_clean = []
    sig_clean = []
    indicators = []
    for iseg, seg in enumerate(sig):
        times_psd, sig_psd, ind_psd = remove_noise_with_psd(times[iseg], seg, fs, window=window)
        
        if len(times_psd) > 0:
            times_clean.extend(times_psd)
            sig_clean.extend(sig_psd)
        indicators.append(ind_psd)
    
    if len(times_clean) > 0:
        times_clean, sig_clean, _ = remove_disconnection(times_clean, sig_clean, fs, stat=[])
        
    return times_clean, sig_clean, indicators

def remove_noise_smooth_unwrap(times_, sig_, indicators_, fs, add_condition=False):

     indicators_    = np.array(indicators_)
     sep = np.where((indicators_[1:]-indicators_[:-1]) != 0)[0] + 1
     segs = np.split(indicators_, sep)
     new_indicators = []
     if len(segs) > 1:
          if np.sum(segs[0]) < 10*fs: 
              segs[0] = np.zeros((len(segs[0]), )) 
          new_indicators.extend(segs[0])
          for i in range(1, len(segs)-1):
              neighbor_pre = np.sum(segs[i-1]) 
              neighbor_post = np.sum(segs[i+1])
              id_min = max(0, i-10)
              id_max = min(len(segs), i + 10)
              is_alone = 0
              for k in range(id_min, id_max):
                  is_alone += np.sum(segs[k])
              if neighbor_pre >= 10*fs and neighbor_post >= 10*fs and len(segs[i]) < 6*fs:
                  segs[i] = np.ones((len(segs[i]), ))
              if neighbor_pre + neighbor_post >= 20*fs and len(segs[i]) < 6*fs:
                  segs[i] = np.ones((len(segs[i]), ))
              if is_alone==0 and add_condition:
                  segs[i] = np.zeros((len(segs[i]), )) 
              if np.sum(segs[i]) < 10*fs: 
                  segs[i] = np.zeros((len(segs[i]), )) 
              new_indicators.extend(segs[i])
          if np.sum(segs[-1]) < 10*fs: 
              segs[-1] = np.zeros((len(segs[-1]), )) 
          new_indicators.extend(segs[-1])
     else:
          new_indicators.extend(segs[0])
     
     # indicators_ = np.array(new_indicators)
     
     # # % REMOVE LONELY VALID SEGMENTS
     # sep = np.where((indicators_[1:]-indicators_[:-1]) != 0)[0] + 1
     # segs = np.split(indicators_, sep)
     # new_indicators = []
     # if len(segs) > 4:
     #      new_indicators.extend(segs[0])
     #      new_indicators.extend(segs[1])
     #      new_indicators.extend(segs[-2])
     #      new_indicators.extend(segs[-1])
     #      for i in range(2, len(segs)-2):
     #          neighbor_pre  = np.sum(segs[i-2]) 
     #          neighbor_post = np.sum(segs[i+2])
     #          current       = np.sum(segs[i])
              
     #          length_pre    = len(segs[i-1]) 
     #          length_post   = len(segs[i+1])
              
     #          if neighbor_pre >= 10*fs and neighbor_post >= 10*fs and\
     #              length_pre <= 10*fs and length_post <= 10*fs and current >= 10*fs:
     #                  segs[i] = np.ones((len(segs[i]), ))
     #          else:
     #              segs[i] = np.zeros((len(segs[i]), ))
     #          new_indicators.extend(segs[i])
         
     # else:
     #      for i in range(len(segs)):
     #          new_indicators.extend(segs[i])
     
     new_indicators = np.array(new_indicators)
     
     times_clean = []
     sig_clean = []
     if len(new_indicators)!=0:
         imin = 0
         for i in range(len(times_)):
             imax = imin + len(times_[i])
             ind_seg = new_indicators[imin:imax]
             if len(times_[i][ind_seg > 0])!=0:
                 times_clean.append(times_[i][ind_seg > 0])
                 sig_clean.append(sig_[i][ind_seg > 0])
             imin = imax
     
             
     if len(times_clean) > 0:
         times_clean, sig_clean, _ = remove_disconnection(unwrap(times_clean), 
                                                          unwrap(sig_clean), 
                                                          fs, stat=[])
     
     return times_clean, sig_clean, new_indicators

def remove_false_rr(rr):
     
    Q1 = np.percentile(rr, 25, interpolation = 'midpoint') 
  
    Q3 = np.percentile(rr, 75, interpolation = 'midpoint') 
    
    qd = (Q3 - Q1) / 2
    
    maximum_expected_diff = 3.32*qd
    minimal_artifact_diff = np.median(rr) - 2.9*qd
    
    criterion = (maximum_expected_diff + minimal_artifact_diff)/2
    
    clean_rr = []
    for i in range(len(rr)):
         if rr[i] > criterion:
              rr[i] = np.mean(rr)

    return clean_rr

def remove_outliers_rr(rr):

    mean_ = np.mean(rr)
    cut_off = 2 * np.std(rr)
    lower_limit = mean_ - cut_off
    upper_limit = mean_ + cut_off
        
    for i in range(len(rr)):
        if rr[i] <= lower_limit and rr[i] >= upper_limit:
            rr[i] = (rr[i+1]+rr[i-1])/2
           
    return rr


def remove_noise_grenoble(sig, fs=256, threshold=50000000, threshold_std=1E6,
                          sec=1, min_clean_window=2):
    ''' Remove saturation using the derivative of the signal and
    flat parts of signal

    Parameters
    ---------------------------
    time_ : timestamp
    signal_ :array signal
    fs: sampling frequency
    threshold: remove signal where abs(derivative) > threshold
    sec: cut sample around the threshold
    min_clean_window: minimum time of clean signal per window (seconds)
    threshold_std: remove signal where std(derivative) > threshold_std

    Returns
    ---------------------------s
    Signal without noise

    '''
    sig = np.array(sig)
    x_s = np.linspace(0, len(sig)/fs, len(sig))
    window = round(fs*sec)
    sig_deriv = np.array((sig[1:]-sig[0:-1])/((x_s[1:]-x_s[0:-1])))
    std_ = []
    for i in range(len(sig_deriv)-fs):
        if i < fs:
            std_.append(np.std(sig_deriv[i:i+2*fs]))
        elif i > len(sig_deriv)-fs:
            std_.append(np.std(sig_deriv[i-2*fs:]))

        else:
            std_.append(np.std(sig_deriv[i-fs:i+fs]))
    for i in range(len(sig_deriv)-fs, len(sig_deriv)):
        std_.append(np.std(sig_deriv[i-fs:]))
    idx_nn_sat = np.where(sig_deriv < threshold)[0]
    idx_nn_flat = np.where(np.array(std_) > threshold_std)[0]
    idx_clean = np.intersect1d(idx_nn_sat, idx_nn_flat)
    indicators = np.zeros((len(sig),))
    sep = np.where((idx_clean[1:]-idx_clean[:-1]) != 1)[0]

    window_clean_sig = min_clean_window*fs

    if sep.tolist():
        if len(idx_clean[0:max(sep[0]-window, 0)]) > window_clean_sig:
            ids = idx_clean[0:max(sep[0]-window, 0)]
            indicators[ids] = 1
        for j in range(1, len(sep)):
            if len(sig[idx_clean[sep[j-1]+1]+window:
                       max(idx_clean[sep[j]]-window, 0)]) > window_clean_sig:
                imin = round((int((idx_clean[sep[j-1]+1]+window)/fs)+1)*fs)
                imax = round(max(idx_clean[sep[j]]-window, 0))
                indicators[imin:imax] = 1
        if len(sig[idx_clean[sep[-1]+1]+window:]) > window_clean_sig:
            imin = round((int((idx_clean[sep[-1]+1]+window)/fs)+1)*fs)            
            indicators[imin:] = 1
    else:
        if len(sig[idx_clean]) > window_clean_sig:
            imin = round((int(idx_clean[0]/fs)+1)*fs)
            indicators[idx_clean] = 1
    return indicators

def remove_noise_still(times_, sig_, indicators_clean, still_times, move_times):
    new_indicators_clean    = indicators_clean.copy()
    times_clean             = []
    sig_clean               = []
    
    if still_times is not None:
        for iseg, times in enumerate(times_):
            if len(still_times) == 0:
                continue
            for still_time in still_times:
                idx     = np.where(times >= still_time)[0]
                if len(idx) == 0:
                    continue
                imin = idx[0]
                imax = len(times)
                
                if len(move_times) > 0:
                    idx = np.where(move_times >= still_time)[0]
                    if len(idx) > 0:
                        move_time = move_times[idx[0]]
                        idx = np.where(times >= move_time)[0]
                        if len(idx) > 0:
                            imax = idx[0]
                    
                new_indicators_clean[iseg][imin:imax] = 0
    
        for iseg, times in enumerate(times_):
            sig = sig_[iseg]
            ind1 = np.where(new_indicators_clean[iseg] == 1)[0]
            if len(ind1) > 0: 
                times_clean.append(times[ind1])
                sig_clean.append(sig[ind1])
            
    return times_clean, sig_clean, new_indicators_clean


def remove_noise_peak_valley(times, sig, fs, peaks, amps,
                             window_time, n_peak_min, period_max, strict=True):
    window              = int(window_time*fs) 
    peaks = np.array(peaks)
    amps  = np.array(amps)
    times_clean         = []
    sig_clean           = []
    
    indicators          = np.zeros(len(sig))
    imax = 0
    
    #remove segment smaller than window 
    if len(sig) > window:
        indicators  = np.ones(len(sig))
        for i in range(0, len(sig), window):
            imin    = i
            imax    = imin + window
            seg     = sig[imin:imax]
            IMIN    = max([0, imin-fs])
            IMAX    = min([len(sig), imax+fs])
            

            #the last small segment gets the same indicator as the precedent indicator
            if imax > len(sig) and len(seg) < window:
                last_indicator = indicators[imin-1]
                imin_ = imin #imax #### changed from imax to imin here else it made no sens
                imax_ = len(sig)
                if last_indicator == 0:

                    indicators[imin_:imax_] = 0
                continue
            
            ### get peaks and amplitude in the segment
            #check is there is at least one peak in or after this segment
            iminp   = np.where(peaks >= imin)[0]
            if len(iminp) == 0:
                indicators[IMIN:IMAX] = 0
                continue

            #check is there is at least one peak in or before this segment
            imaxp         = np.where(peaks <= imax)[0]

            if len(imaxp) == 0:
                indicators[IMIN:IMAX] = 0
                continue
                      
            iminp   = iminp[0]
            imaxp   = imaxp[-1]
            amp_rs  = amps[iminp:imaxp]
            peaks_rs = peaks[iminp:imaxp]
    
            # Remove noise when peaks distance is not stable (at least n_peak_min peaks needed for window_time segment)
            if len(peaks_rs) <= n_peak_min:
                indicators[IMIN:IMAX] = 0
                continue
            # Remove noise when peaks distance is not stable (time intervalle between peaks > period_max seconds)
            is_rr_stable = True
            for ip in range(len(peaks_rs)):
                t = (times[peaks_rs[ip]] - times[peaks_rs[ip-1]]) / np.timedelta64(1, 's')
                if t > period_max:
                    is_rr_stable = False
            
            t = (times[peaks_rs[1]] - times[peaks_rs[0]]) / np.timedelta64(1, 's')
            if t > period_max:
                is_rr_stable = False
                    
            if not is_rr_stable:
                indicators[IMIN:IMAX] = 0
                continue
            
            # Remove noise when successive peak amplitude ratio is not stable
            amp_rs_var      = 100*abs((abs(amp_rs[1:]) - abs(amp_rs[:-1]))/ abs(amp_rs[:-1]))
            amp_rs_high_var1 = amp_rs_var[amp_rs_var > 150]
            amp_rs_high_var2 = amp_rs_var[amp_rs_var > 75]
            if len(amp_rs_high_var1) >= 1 and len(amp_rs_high_var2) >= 3:
                indicators[IMIN:IMAX] = 0

            if not strict:            
                if np.std(sig[peaks_rs]) < 8:
                    indicators[IMIN:IMAX] = 1
                    
        times_clean = times[indicators == 1]
        sig_clean   = sig[indicators == 1]
        
    return times_clean, sig_clean, indicators

def remove_noise_peak_valley_unwrap(times, sig, fs, peaks, amps,
                                    window_time, n_peak_min, period_max, strict=True):
    sig_clean           = []
    times_clean         = []
    indicators_clean    = []
    for id_seg, seg in enumerate(sig):

        times_seg   = times[id_seg]
        peaks_seg   = peaks[id_seg]
        amps_seg    = amps[id_seg] 
        t_clean, s_clean, i_clean = remove_noise_peak_valley(times_seg, seg, fs, peaks_seg, amps_seg,
                                                             window_time, n_peak_min, period_max, strict=strict)

        indicators_clean.append(i_clean)
        if len(t_clean) > 0:
            sig_clean.extend(s_clean)
            times_clean.extend(t_clean)

    times_clean, sig_clean, _ = remove_disconnection(times_clean, sig_clean, fs)
    
    return times_clean, sig_clean, indicators_clean






def method_window_np(sig, window_s, fs=200, method = 'max'):
    """
    apply methods on consecutive windows and return an array of same size as sig with the results
    
    """
    window           = int(fs*window_s)
    
    nb_windows       =  int(len(sig)/window)
    sigtemp  = np.array(sig)
    temp_val = np.array(sig)
   
   
    if method == 'min':
        #for all segments of window_s get min over window_s 
        for i in range( nb_windows ):
            temp_val[i * window :(i+1) * window ] =  min(sigtemp[i * window : (i+1) * window])
        #fill the end that lasts less than a minute 
        if (len(sig)> nb_windows * window):
            endmed = min( sigtemp[(nb_windows * window +1) :])
            temp_val[nb_windows * window:len(sig)-1] = endmed
            
    if method == 'max':
        #for all segments of window_s get max over window_s 
        for i in range( nb_windows ):
            temp_val[i * window :(i+1) * window ] =  max(sigtemp[i * window : (i+1) * window])
        #fill the end that lasts less than a minute 
        if (len(sig)> nb_windows * window):
            endmed = max( sigtemp[(nb_windows * window +1) :])
            temp_val[nb_windows * window:len(sig)-1] = endmed
            
    if method == 'sum':
         #for all segments of window_s get sum over window_s 
         for i in range( nb_windows ):
             temp_val[i * window :(i+1) * window ] =  sum(sigtemp[i * window : (i+1) * window])
         #fill the end that lasts less than a minute 
         if (len(sig)> nb_windows * window):
             endmed = sum( sigtemp[(nb_windows * window +1) :])
             temp_val[nb_windows * window:len(sig)-1] = endmed
             
    if method == 'mean':
         #for all segments of window_s get mean over window_s  
         for i in range( nb_windows ):
             temp_val[i * window :(i+1) * window ] =  np.mean(sigtemp[i * window : (i+1) * window])
         #fill the end that lasts less than a minute 
         if (len(sig)> nb_windows * window):
             endmed = np.mean( sigtemp[(nb_windows * window +1) :])
             temp_val[nb_windows * window:len(sig)-1] = endmed
            
    return temp_val

def remove_no_rsp_signal(
        rsp_time,
        rsp_filt, 
        fs = 20,
        window_s = 15, 
        rsp_amp_min = 6):
    
    df = pd.DataFrame({'rsp_f' : rsp_filt, 'rsp_t' : rsp_time})
    df['no_breath'] = 0
    
    # Minimum breath (rsp) amplitude on filtered signal
    df['max_f'] =  method_window_np(sig = df['rsp_f'].values, 
                                    window_s = window_s, fs=fs, method = 'max')
    df['min_f'] =  method_window_np(sig = df['rsp_f'].values, 
                                    window_s = window_s, fs=fs, method = 'min')
    
    # Compute amplitude of windows 
    df['amp'] = df['max_f'] - df['min_f']
    df.loc[df['amp'] < rsp_amp_min, 'no_breath' ] = 1
    df['no_breath_sum'] =  method_window_np(sig = df['no_breath'].values, 
                                     window_s = window_s, fs=fs, method = 'sum')
    
    datagood = df.loc[df['no_breath_sum']==0]
    
    indicateur  = df['no_breath_sum']==0
    indicateur = [int(ind)*100 for ind in indicateur]
    
    return datagood['rsp_t'].values, datagood['rsp_f'].values, indicateur

def remove_no_rsp_signal_unwrap(
        rsp_time, 
        rsp_filt, 
        fs = 20, 
        window_s = 15, 
        rsp_amp_min = 6
        ):
    
    sig_clean           = []
    times_clean         = []
    indicators_clean    = []
    for id_seg, seg in enumerate(rsp_filt):
        t_clean, s_clean, i_clean = remove_no_rsp_signal(
                                    rsp_time = rsp_time[id_seg],
                                    rsp_filt = seg,
                                    window_s = window_s, 
                                    rsp_amp_min = rsp_amp_min)

        indicators_clean.append(i_clean)
        if len(t_clean) > 0:
            sig_clean.extend(s_clean)
            times_clean.extend(t_clean)

    times_clean, sig_clean, _ = remove_disconnection_loss(times_clean, sig_clean, fs)
    
    return times_clean, sig_clean, indicators_clean


def remove_saturation_and_big_ampls (ecg_raw, ecg_filt, ecg_time, 
                  window_s = 5, 
                  sat_high = 3750,sat_low = 50,
                  amp_max = 1000, amp_min = 20):
    
    
    df = pd.DataFrame({'ecg_r':ecg_raw, 'ecg_f':ecg_filt, 'ecg_t':ecg_time})
    
    #saturations on raw signal
    df['saturation']  = 0
    df.loc[df['ecg_r']>sat_high, 'saturation'] = 1
    df.loc[df['ecg_r']<sat_low, 'saturation'] = 1
    
    #amplitude on filtered signal
    df['max_f'] =  method_window_np(sig = df['ecg_f'].values, window_s = window_s, fs=200, method = 'max')
    df['min_f'] =  method_window_np(sig = df['ecg_f'].values, window_s = window_s, fs=200, method = 'min')
    df['amp'] = df['max_f'] - df['min_f']

    df.loc[df['amp']>amp_max,'saturation' ] =1
    df.loc[df['amp']<amp_min,'saturation' ] =1
    
    df['satsum'] =  method_window_np(sig = df['saturation'].values, window_s = window_s, fs=200, method = 'sum')
    
    datagood = df.loc[df['satsum']==0]
    
    indicateur  = df['satsum']==0
    indicateur = [int(ind) for ind in indicateur]
    
    return datagood['ecg_t'].values, datagood['ecg_f'].values, indicateur

def remove_saturation_and_big_ampls_unwrap(ecg_raw, ecg_filt, ecg_time, fs = 200, 
                  window_s = 5, 
                  sat_high = 3750,sat_low = 50,
                  amp_max = 1000, amp_min = 20):
    sig_clean           = []
    times_clean         = []
    indicators_clean    = []
    for id_seg, seg in enumerate(ecg_filt):

        t_clean, s_clean, i_clean = remove_saturation_and_big_ampls(
                                        ecg_raw = ecg_raw[id_seg] ,
                                        ecg_filt = seg,
                                        ecg_time = ecg_time[id_seg], 
                                        window_s = window_s, 
                                        sat_high = sat_high,
                                        sat_low = sat_low ,
                                        amp_max = amp_max,
                                        amp_min = amp_min)

        indicators_clean.append(i_clean)
        if len(t_clean) > 0:
            sig_clean.extend(s_clean)
            times_clean.extend(t_clean)

    times_clean, sig_clean, _ = remove_disconnection_loss(times_clean, sig_clean, fs)
    
    return times_clean, sig_clean, indicators_clean