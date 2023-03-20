import numpy as np

#from pylife.useful import unwrap
from pylife.env import get_env
from pylife.filters import low_pass_filter, low_high_pass_f
from pylife.filters import Butterlife
from pylife.detect_breath_peaks import rsp_findpeaks
from pylife.ecg_derived_resp import extract_edr
#from pylife.useful import unwrap
from pylife.activity_measurement import signal_magnetude_area
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import matplotlib.pyplot as plt
# --- Add imports for PROD and DEV env

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC
    /blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind,
                          np.unique(np.hstack((indnan,
                                               indnan-1,
                                               indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

def detect_peaks_unwrap(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):
    peaks_s = []
    amps_s = []
    for i, seg in enumerate(x):
        peaks   = detect_peaks(seg, mph, mpd, threshold, edge, kpsh, valley)
        amps    = np.array(seg)[peaks]
        peaks_s.append(peaks)
        amps_s.append(amps)

    return peaks_s, amps_s

# NEW function inspired from neurokit detection peaks for BREATH !!!
def detect_rsp_peaks(seg, fs):
    peaks = [] 
    troughs = []
    
    # Verify if segment ok
    seg = np.atleast_1d(seg).astype('float64')
    if seg.size < 30:
        peaks =  np.array([], dtype=int)
        troughs =  np.array([], dtype=int)
        return peaks, troughs 
    
    # Extract peaks
    peaks_dict = rsp_findpeaks(seg, fs) 
    peaks = peaks_dict['RSP_Peaks']
    troughs = peaks_dict['RSP_Troughs']
    
    return peaks, troughs

# NEW function inspired from neurokit detection peaks for BREATH !!!
def detect_rsp_peaks_unwrap(x, fs):
    peaks_s = []
    troughs_s = []
    peaks_amps_s = []
    troughs_amps_s = []
    
    for i, seg in enumerate(x):
        peaks, troughs = detect_rsp_peaks(seg, fs)
        amps_peak = np.array(seg)[peaks]
        amps_trough = np.array(seg)[troughs]
        
        peaks_s.append(peaks)
        troughs_s.append(troughs)
        peaks_amps_s.append(amps_peak)
        troughs_amps_s.append(amps_trough)
        
    return peaks_s, troughs_s, peaks_amps_s, troughs_amps_s 

def integrate_trapz(sig):
    """Integration  using trapezoidal rule
    """
    s = 0
    for i in range(1, len(sig)):
        s += (sig[i-1]+sig[i])/2
    return s


def detect_qrs(sig, fs=250, window=0.15, mpd=int(0.55*200), rm_peaks=False):
    """ Detect qrs using detect_peaks function

    Parameters
    ----------
    sig: input ecg
    fs
    window: window of integration
    mpd: minimum distance between peaks for peak detection (seconds)

    Returns
    ----------
    peaks: QRS positions

    """
    
    if len(sig) < 1*fs:
        return []
    
    sig = np.array(sig)
    
    s_f = low_high_pass_f(sig, fs, filter_order=2, w_c=15, type_f='low')\
        - low_high_pass_f(sig, fs, filter_order=2, w_c=5, type_f='low')

    x_s_f = np.linspace(0, len(s_f)/fs, len(s_f))
    der = (s_f[1:]-s_f[0:-1])/((x_s_f[1:]-x_s_f[0:-1]))
    a = der*der

    window = int(window*fs)
    inte = []
    for k in range(len(a)-window):
        inte.append(integrate_trapz(a[k:k+window]))
    for k in range(len(a)-window, len(a)):
        inte.append(integrate_trapz(a[k:]))
    intef = low_high_pass_f(inte, fs, filter_order=4, w_c=5, type_f='low')
    intef = low_high_pass_f(intef, fs, filter_order=4, w_c=.5, type_f='high')
    peaks = detect_peaks(intef, mph=0, mpd=mpd, threshold=0, edge='rising',
                         kpsh=False, valley=False)
    peaks = np.array(peaks)

    # Keep valid peaks
    if rm_peaks:
        peaks = detect_valid_peaks_qrs(intef, fs, peaks)
    return peaks

def detect_valid_peaks_qrs(sig, fs, peaks):
    # % REMOVE VERY CLOSE PEAKS
    new_peaks = []
    bad_peak = -1
    for i in range(len(peaks)-1):
        if peaks[i] == bad_peak:
            continue
        if (peaks[i+1] - peaks[i]) < .15*fs:
            peaks_i = [peaks[i+1], peaks[i]]
            ip      = np.argmax([sig[peaks[i+1]], sig[peaks[i]]])
            
            ibad    = np.argmin([sig[peaks[i+1]], sig[peaks[i]]])
            bad_peak = peaks_i[ibad]
            if bad_peak in new_peaks:
                idx = np.where(new_peaks == bad_peak)[0][0]
                new_peaks = np.delete(new_peaks, [idx]).tolist()
                
            new_peaks.append(peaks_i[ip])
        
        else:
            new_peaks.append(peaks[i])
            
    new_peaks.append(peaks[-1])
    peaks = np.unique(new_peaks)
    
    # % REMOVE VERY LOW PEAKS
    new_peaks       = []
    n_neighbours    = 2
    for i in range(0, len(peaks)):
        peak        = peaks[i]
        if i < n_neighbours:
            peaks_i     = peaks[: i+n_neighbours+1]
        elif i > len(peaks) - n_neighbours:
            peaks_i     = peaks[i-n_neighbours:i]
        else:
            peaks_i     = peaks[i-n_neighbours : i+n_neighbours+1]
        intef_peaks_i = sig[peaks_i]
        intef_peaks_i = np.sort(intef_peaks_i)
        vhigh       = np.percentile(intef_peaks_i, 80)
        
        if sig[peak] / vhigh > .1:
            new_peaks.append(peak)
    peaks = np.unique(new_peaks)
    
    return peaks

def detect_valid_peaks_r(sig, fs, peaks):
    # % REMOVE VERY CLOSE PEAKS
    new_peaks   = []
    bad_peak = -1
    for i in range(len(peaks)-1):
        if peaks[i] == bad_peak:
            continue
        if (peaks[i+1] - peaks[i]) < .15*fs:
            peaks_i = [peaks[i+1], peaks[i]]
            ip      = np.argmin([abs(sig[peaks[i+1]]), abs(sig[peaks[i]])])
            ibad    = np.argmin([sig[peaks[i+1]], sig[peaks[i]]])
            bad_peak = peaks_i[ibad]
            if bad_peak in new_peaks:
                idx = np.where(new_peaks == bad_peak)[0][0]
                new_peaks = np.delete(new_peaks, [idx]).tolist()
                
            new_peaks.append(peaks[ip])
        else:
            new_peaks.append(peaks[i])
    
    new_peaks.append(peaks[-1])
    peaks = np.unique(new_peaks)
    
    # % REMOVE VERY LOW PEAKS
    new_peaks       = []
    n_neighbours    = 2
    for i in range(0, len(peaks)):
        peak        = peaks[i]
        if i < n_neighbours:
            peaks_i     = peaks[: i+n_neighbours+1]
        elif i > len(peaks) - n_neighbours:
            peaks_i     = peaks[i-n_neighbours:i]
        else:
            peaks_i     = peaks[i-n_neighbours : i+n_neighbours+1]
        intef_peaks_i = sig[peaks_i]
        intef_peaks_i = np.sort(intef_peaks_i)
        vhigh       = np.percentile(intef_peaks_i, 80)
        
        if sig[peak] / vhigh > .1:
            new_peaks.append(peak)
    peaks = np.unique(new_peaks)
    
    return peaks
    
def detect_valid_peaks_breath_unwrap(sig, fs, peaks):
    peaks_  = []
    amps_   = []
    for i, seg in enumerate(sig):
        peaks_seg = peaks[i]
        if isinstance(peaks_seg, np.ndarray):
            peak, amp = detect_valid_peaks_breath(seg, fs, peaks_seg)
            peaks_.append(peak)
            amps_.append(amp)
        else:
            peaks_.append([])
            amps_.append([])

    return peaks_, amps_

def detect_valid_peaks_breath(sig, fs, peaks):
    # % REMOVE VERY CLOSE PEAKS
    new_peaks   = []
    new_amps    = []
    bad_peak = -1
    if len(peaks) == 0:
        return new_peaks, new_amps
    
    for i in range(len(peaks)-1):
        if peaks[i] == bad_peak:
            continue
        if (peaks[i+1] - peaks[i]) < 2.5*fs:
            peaks_i = [peaks[i+1], peaks[i]]
            ip      = np.argmax([sig[peaks[i+1]], sig[peaks[i]]])
            ibad    = np.argmin([sig[peaks[i+1]], sig[peaks[i]]])
            bad_peak = peaks_i[ibad]
            if bad_peak in new_peaks:
                idx = np.where(new_peaks == bad_peak)[0][0]
                new_peaks = np.delete(new_peaks, [idx]).tolist()
                
            new_peaks.append(peaks_i[ip])
        else:
            new_peaks.append(peaks[i])
    new_peaks.append(peaks[-1])
    
    peaks = np.unique(new_peaks)
    
    # % REMOVE VERY LOW PEAKS
    new_peaks       = []
    new_amps        = []
    n_neighbours    = 2
    for i in range(0, len(peaks)):
        peak        = peaks[i]
        if i < n_neighbours:
            peaks_i     = peaks[: i+n_neighbours]
        elif i > len(peaks) - n_neighbours:
            peaks_i     = peaks[i-n_neighbours:i]
        else:
            peaks_i     = peaks[i-n_neighbours : i+n_neighbours]
        intef_peaks_i = sig[peaks_i]
        intef_peaks_i = np.sort(intef_peaks_i)
        vhigh       = np.percentile(intef_peaks_i, 75)
        
        if sig[peak] / vhigh > .15:
            new_peaks.append(peak)
           
    new_peaks = np.unique(new_peaks)
    new_amps = sig[new_peaks]
    
    return new_peaks, new_amps
    
def detect_peaks_r_unwrap(sig, fs, peaks_qrs):
    peaks_r_ = []
    amplitude_ = []
    for i, seg in enumerate(sig):
        peaks_qrs_seg = peaks_qrs[i]
        peaks_r, amp = detect_peaks_r(seg, fs, peaks_qrs_seg)
        peaks_r_.append(peaks_r)
        amplitude_.append(amp)

    return peaks_r_, amplitude_

def detect_peaks_r(sig, fs, peaks):
    
    sig         = np.array(sig)
    N           = 60*fs
    if len(sig) < 60*fs:
        N           = len(sig)
    if detect_ecg_reversed(sig, fs, N=N):
        sig = -sig
    
    peaks_r    = []
    amplitude   = []
    dec = int(0.2*fs)
    for peak in peaks:
        seg         = sig[peak:peak+dec]
        imax = np.argmax(abs(seg))
        if seg[imax] > 0:
            imin = np.argmin(seg)
            imax = np.argmax(seg)
        else:
            imin = np.argmax(seg)
            imax = np.argmin(seg)
        peak_r = peak + imax
        peaks_r.append(peak_r)
        amplitude.append(seg[imax]-seg[imin])
              
    return np.array(peaks_r), np.array(amplitude)

def detect_qrs_brassiere(sig, fs=200, window=0.15):
    """ Detect qrs using detect_peaks function

    Parameters
    ----------
    sig: input ecg
    fs
    window: window of integration
    mpd: minimum distance between peaks for peak detection (seconds)

    Returns
    ----------al
    peaks: QRS positions

    """
    s_f = low_pass_filter(np.array(sig), cut_off=15, fs=fs)\
        - low_pass_filter(np.array(sig), cut_off=5, fs=fs)

    x_s_f = np.linspace(0, len(s_f)/fs, len(s_f))
    der = (s_f[1:]-s_f[0:-1])/((x_s_f[1:]-x_s_f[0:-1]))

    a = der*der
    window = int(window*fs)
    inte = []
    for k in range(len(a)-window):
        inte.append(integrate_trapz(a[k:k+window]))
    for k in range(len(a)-window, len(a)):
        inte.append(integrate_trapz(a[k:]))

    butter = Butterlife()
    intef = butter.filt(inte, fs, fc=10, order=2, ftype='low')

    peaks = detect_peaks(intef, mph=0.25E9, mpd=0, threshold=0, edge='rising',
                         kpsh=False, valley=False)

    peaks = np.array(peaks)

    # Keep valid peaks
    keep_peaks = []
    if len(peaks) > 1:
        intef = intef/max(abs(intef))
        peak_values = intef[peaks]
        peaks_value_ref = np.percentile(peak_values, 65)

        keep_peaks = []
        for ipeak in peaks:
            peak_value = intef[ipeak]
            if peak_value/peaks_value_ref > 0.3:
                keep_peaks.append(ipeak)

    return np.array(keep_peaks)


def detect_qrs_unwrap(sig, fs=250, window=0.15, mpd=int(0.55*200), rm_peaks=False):

    peaks_s = []

    for id_seg, seg in enumerate(sig):
        peaks = detect_qrs(seg, fs, window=window, mpd=mpd, rm_peaks=rm_peaks)
        peaks_s.append(peaks)

    return peaks_s


def detect_qrs_brassiere_unwrap(sig, fs=200):

    peaks_s = []

    for id_seg, seg in enumerate(sig):
        peaks = detect_qrs_brassiere(seg, fs)
        peaks_s.append(peaks)

    return peaks_s


def detect_qrs_amplitude(sig, fs, show_plot=False):
    """ Detect qrs amplitude of one array ecg or a list of arrays ecg
    Input: sample, name, fe, show_plot (sample must be > 35 samples)
    sig: input ecg
    fe : sampling rate
    Output: points_resp, np.mean(stat_amp), np.std(stat_amp)
    points_resp : maxs that can be used to extract respiration from the ecg
    np.mean(stat_amp) : mean amplitude of the QRS
    np.std(stat_amp) : std amplitude of the QRS
    """
    stat_amp = []

    peaks = detect_qrs(sig, fs, window=0.15)
    points_min = []
    points_max = []
    amp = []
    offset = 60
    dec = 2
    for k in range(1, len(peaks)):
        max_ = dec+peaks[k]-offset + np.argmax(sig[dec+peaks[k]-offset:
                                                   dec+peaks[k]+offset])
        min_ = dec+peaks[k]-offset + np.argmin(sig[dec+peaks[k]-offset:
                                                   dec+peaks[k]+offset])
        points_min.append(min_)
        points_max.append(max_)
        amp.append(sig[max_]-sig[min_])
    stat_amp.extend(np.array(amp) * 2.4 * 1000 / (4095 * 824))
    if DEV:
        if show_plot:
            data_time = np.linspace(0, len(sig)/fs, len(sig))
            plt.figure()
            plt.plot(data_time, sig)
            for j in points_min:
                plt.scatter(data_time[j], sig[j], c='r')
            for j in points_max:
                plt.scatter(data_time[j], sig[j], c='b')

    return points_max, stat_amp


def detect_qrs_mpd_update_unwrap(sig, fs, window_time=20, mpd_init=None):

    peaks_s = []
    mpds_s = []
    for id_seg, seg in enumerate(sig):
        peaks, mpds = detect_qrs_mpd_update(seg, fs, window_time=window_time,
                                            mpd_init=mpd_init)
        peaks_s.append(peaks)
        mpds_s.append(mpds)

    return peaks_s, mpds_s


def detect_qrs_mpd_update(sig, fs, window_time=20, mpd_init=None):
    """ Detect qrs by updating mpd parameter every "window_time" seconds
    according to RR interval mean value

    Parameters
    ----------
    sig: input ecg
    fs: sampling frequency
    window_time: time window for peaks detection
    mpd_init: minimum distance between peaks
    for peak detection on first ecg segment (seconds)

    Returns
    ----------
    peaks: QRS positions

    """
    # initialization
    mpd_s = []
    window_size = round(window_time*fs)

    # If mpd_init is not defined, the value is automatically calculated
    # using the mean heart rate at the beginning of the signal
    if mpd_init is None:
        mpd_init = 0.5  # arbitrary value
        if len(sig) > window_size:
            seg_init = sig[:window_size]
            peaks_init = detect_qrs(seg_init, fs=fs, window=0.15, mpd=mpd_init)
            if len(peaks_init) > 1:
                # Heart Rate (hr)
                hr_init = np.mean((peaks_init[1:] - peaks_init[:-1])/fs)
                mpd_init = hr_init*2/3  # arbitrary value
    mpd = mpd_init
    mpd_s.append(mpd)

    # offset is created to avoid miss detection at the ends of the segment
    offset = round(1*fs)
    if window_size + 2*offset > len(sig):
        peaks_s = detect_qrs(sig, fs=fs, window=0.15, mpd=mpd)
    else:
        peaks_s = []
        imin = 0
        imax = len(sig)

        for imin_w in range(imin, imax-window_size, window_size):

            imax_w = imin_w + window_size

            if (imin_w - offset) >= 0 and (imax_w + offset) <= imax:
                peaks = detect_qrs(sig[imin_w-offset:imax_w], fs=fs,
                                   window=0.15, mpd=mpd)
                peaks = peaks + (imin_w-offset)
            elif (imin_w - offset) < 0 and (imax_w + offset) <= imax:
                peaks = detect_qrs(sig[imin_w:imax_w+offset], fs=fs,
                                   window=0.15, mpd=mpd)
                peaks = peaks + (imin_w)
            if (imin_w - offset) >= 0 and (imax_w + offset) > imax:
                peaks = detect_qrs(sig[imin_w-offset:imax_w], fs=fs,
                                   window=0.15, mpd=mpd)
                peaks = peaks + (imin_w-offset)

            peaks = np.array(peaks)
            if len(peaks) > 1:

                # Keep peaks in the window of interest (without offset)
                ipeaks_min = np.where(peaks >= imin_w)
                ipeaks_max = np.where(peaks < imax_w)
                ipeaks = np.intersect1d(ipeaks_min, ipeaks_max)
                peaks = peaks[ipeaks]
                peaks_s.extend(peaks)
                # Heart Rate (hr)
                hr = np.mean((peaks[1:] - peaks[:-1])/fs)
                # Cap heart rate betwween min and max value
                hr = max(hr, 60/300)  # Max 300 bpm
                hr = min(hr, 60/40)  # Min 40 bpm
                mpd = hr*2/3  # arbitrary value
                mpd_s.append(mpd)

        # Detect peaks at the end of the signal
        peaks = detect_qrs(sig[imax_w-offset:], fs=fs, window=0.15, mpd=mpd)
        peaks = peaks + (imax_w-offset)
        ipeaks_min = np.where(peaks >= imax_w)
        ipeaks_max = np.where(peaks < imax)
        ipeaks = np.intersect1d(ipeaks_min, ipeaks_max)
        peaks = peaks[ipeaks]
        peaks_s.extend(peaks)
        peaks_s = np.array(peaks_s)

    return peaks_s, mpd_s


def detect_saturation(times, sig, threshold=3000, fs=200, sec=1):
    """Remove noisy signal. Remove data superior to threshold
    or inferior to -threshold
    Input: ecg, threshold, fe, show_plot
    ecg : input signal
    threshold
    fs: sampling rate
    Output: data
    data : list of non-noisy samples
    """
    window = int(fs*sec)
    sig = np.array(sig)
    times = np.array(times)
    idx_clean = np.where(abs(sig) < threshold)[0]
    indicators = np.zeros((len(sig), 1))
    data = []
    new_time = []
    sep = np.where((idx_clean[1:]-idx_clean[:-1]) != 1)[0]
    if sep.tolist():
        data.append(sig[idx_clean[0+window:max(sep[0]-window, 0)]])
        new_time.append(times[idx_clean[0+window:max(sep[0]-window, 0)]])
        indicators[idx_clean[0+window:max(sep[0]-window, 0)]] = 1
        for i in range(1, len(sep)):
            data.append(sig[idx_clean[sep[i-1]+1]+window:
                            max(idx_clean[sep[i]]-window, 0)])
            new_time.append(times[idx_clean[sep[i-1]+1]+window:
                                  max(idx_clean[sep[i]]-window, 0)])
            indicators[idx_clean[sep[i-1]+1]+window:
                       max(idx_clean[sep[i]]-window, 0)] = 1
        data.append(sig[idx_clean[sep[-1]+1]+window:])
        new_time.append(times[idx_clean[sep[-1]+1]+window:])
        indicators[idx_clean[sep[-1]+1]+window:] = 1
    else:
        new_time.append(times[idx_clean])
        data.append(sig[idx_clean])
        indicators[idx_clean] = 1
    data_ = []
    time_ = []
    for sample in enumerate(data):
        if sample[1].tolist():
            data_.append(sample[1])
            time_.append(new_time[sample[0]])
    return time_, data_, indicators


def detect_breathing(sig, window_peaks=4, amp_min=50, fs=20):
    """ Detect qrs using detect_peaks function

    Parameters
    ----------
    sig: input ecg
    fs
    window: window of integration
    amp_min: minimum amplitude between successive peak and valley
    for peak validation (seconds)

    Returns
    ----------
    peaks: QRS positions

    """
    new_peaks = []
    peaks = detect_peaks(sig, mph=-100000,
                         mpd=0, threshold=0, edge="rising")
    valley = detect_peaks(sig, mph=-10000000000,
                          mpd=0, threshold=0, edge="rising", valley=True)

    if len(peaks) > 2:
        diff = []
        id_peaks = -1
        for id_valley in valley:
            if peaks[peaks > id_valley].tolist():
                id_peaks = np.where(
                    peaks == peaks[peaks > id_valley][0]
                     )[0][0]
                diff.append(abs(sig[peaks[id_peaks]]
                                - sig[id_valley]))

        diff = np.array(diff)
        remove_peaks_to_close_to_valley = []
        if peaks.tolist():
            for id_valley in valley:
                if peaks[peaks > id_valley].tolist():
                    id_peaks = np.where(
                        peaks == peaks[peaks > id_valley][0]
                                        )[0][0]
                if id_peaks > -1:
                    if id_peaks < window_peaks:
                        range_amp = diff[: id_peaks + window_peaks]
                        range_amp = np.median(range_amp[np.argsort(range_amp)[-4:]])
                    elif id_peaks > len(diff) - window_peaks:
                        range_amp = diff[id_peaks - window_peaks:]
                        range_amp = np.median(range_amp[np.argsort(range_amp)[-4:]])
                    else:
                        range_amp = diff[id_peaks-window_peaks:
                                         id_peaks + window_peaks]
                        range_amp = np.median(range_amp[np.argsort(range_amp)[-4:]])
                    if abs(sig[peaks[id_peaks]]
                           - sig[id_valley]) < 65*range_amp/100:
                        remove_peaks_to_close_to_valley.append(peaks[id_peaks])
        true_peaks = []
        for id_peak in peaks:
            if id_peak not in remove_peaks_to_close_to_valley:
                true_peaks.append(id_peak)
        for i in true_peaks:
            id_valley = np.argmin(abs(i-valley))
            d = sig[i] - sig[valley[id_valley]]
            if d > amp_min:
                new_peaks.append(i)
    else:
        new_peaks = peaks

    sig = np.array(sig)
    amplitudes = []
    if len(new_peaks) > 0:
        amp = []
        offset = int(1.5 * fs)
        for k in range(1, len(new_peaks)):
            if len(sig[new_peaks[k]-offset: new_peaks[k]+offset]):
                 max_ = new_peaks[k]-offset + np.argmax(sig[new_peaks[k]-offset:
                                                            new_peaks[k]+offset])
                 min_ = new_peaks[k]-offset + np.argmin(sig[new_peaks[k]-offset:
                                                            new_peaks[k]+offset])
                 amp.append(sig[max_]-sig[min_])
            # else:
                 # print(peaks, k)
        amplitudes.extend(np.array(amp))


  #  xs = np.linspace(0, len(sig)/20, len(sig))
#    plt.figure()
#    plt.plot(xs, sig)
#    for i in peaks:
#        plt.scatter(xs[i], sig[i], c='black')
#    for i in valley:
#        plt.scatter(xs[i], sig[i], c='b')
#    plt.ylabel('AMPLITUDE')
#    plt.xlabel('Number of samples')
#
#    plt.figure()
#    plt.plot(xs, sig)
#    for i in true_peaks:
#        plt.scatter(xs[i], sig[i], c='g')
#    plt.figure()
#    plt.plot(xs, sig)
#    for i in new_peaks:
#        plt.scatter(xs[i], sig[i], c='r')
#    plt.ylabel('AMPLITUDE')
#    plt.xlabel('Number of samples')

    return new_peaks, amplitudes


def detect_breathing2(sig, window_peaks=1, amp_min=50):
    """ Detect qrs using detect_peaks function

    Parameters
    ----------
    sig: input ecg
    fs
    window: window of integration
    amp_min: minimum amplitude between successive peak and valley
    for peak validation (seconds)

    Returns
    ----------
    peaks: QRS positions

    """

    window_peaks = 1
    new_peaks = []
    peaks = detect_peaks(sig, mph=-100000,
                         mpd=0, threshold=0, edge="both")
    valley = detect_peaks(sig, mph=-10000000000,
                          mpd=0, threshold=0, edge="both", valley=True)

    if len(peaks) > 2:
        diff = []
        id_peaks = -1
        for id_valley in valley:
            if peaks[peaks > id_valley].tolist():
                id_peaks = np.where(
                    peaks == peaks[peaks > id_valley][0]
                                    )[0][0]
                diff.append(abs(sig[peaks[id_peaks]]
                                - sig[id_valley]))
        diff = np.array(diff)
        remove_peaks_to_close_to_valley = []
        if peaks.tolist():
            for id_valley in valley:
                if peaks[peaks > id_valley].tolist():
                    id_peaks = np.where(
                        peaks == peaks[peaks > id_valley][0]
                                        )[0][0]
                if id_peaks > -1:
                    if id_peaks < window_peaks:
                        range_amp = diff[: id_peaks + window_peaks]
                        range_amp = np.median(range_amp[np.argsort(range_amp)[-window_peaks:]])
                    elif id_peaks > len(diff) - window_peaks:
                        range_amp = diff[id_peaks - window_peaks:]
                        range_amp = np.median(range_amp[np.argsort(range_amp)[-window_peaks:]])
                    else:
                        range_amp = diff[id_peaks-window_peaks:
                                         id_peaks + window_peaks]
                        range_amp = np.median(range_amp[np.argsort(range_amp)[-window_peaks:]])
                    if abs(sig[peaks[id_peaks]]
                           - sig[id_valley]) < 65*range_amp/100:
                        remove_peaks_to_close_to_valley.append(peaks[id_peaks])
        true_peaks = []
        for id_peak in peaks:
            if id_peak not in remove_peaks_to_close_to_valley:
                true_peaks.append(id_peak)
        pb_val = []
        rm_p = []
        for id_v in range(1, len(valley)-1):
            if sig[valley[id_v]] > sig[valley[id_v + 1]]/1.8 and sig[valley[id_v]] > sig[valley[id_v - 1]]/1.8:
                 pb_val.append(valley[id_v])

                 id_r = np.argmin(abs(peaks[peaks > valley[id_v]]-valley[id_v]))
                 p_r = peaks[peaks > valley[id_v]][id_r]
                 id_l = np.argmin(abs(peaks[peaks < valley[id_v]]-valley[id_v]))
                 p_l = peaks[peaks < valley[id_v]][id_l]
                 if p_r in true_peaks and p_l in true_peaks:
                      rm_p.append(p_l)
        rm_n = []
        pb_val2 = []
        for id_v in range(1, len(valley)-1):
            if sig[valley[id_v]] < 2.5*sig[valley[id_v + 1]] and sig[valley[id_v]] < 2.5*sig[valley[id_v - 1]]:
                 pb_val2.append(valley[id_v])

                 id_r = np.argmin(abs(peaks[peaks > valley[id_v]]-valley[id_v]))
                 p_r = peaks[peaks > valley[id_v]][id_r]
                 id_l = np.argmin(abs(peaks[peaks < valley[id_v]]-valley[id_v]))
                 p_l = peaks[peaks < valley[id_v]][id_l]
                 if p_r in true_peaks and p_l in true_peaks and p_r not in rm_p and p_l not in rm_p:
                      rm_n.append(p_l)

        rm_oscillations = []
        for i in true_peaks:
            id_valley0 = np.argsort(abs(i-valley))[0]
            id_valley1 = np.argsort(abs(i-valley))[1]
            d0 = abs(sig[i] - sig[valley[id_valley0]])
            d1 = abs(sig[i] - sig[valley[id_valley1]])
            if max(d0, d1) < amp_min:
               rm_oscillations.append(i)

        new_peaks = []
        for i in true_peaks:
            if i not in rm_p and i not in rm_oscillations and i not in rm_n:
                 new_peaks.append(i)
    else:
        new_peaks = peaks

#    import matplotlib.pyplot as plt
#    if len(new_peaks) > 3:
#         xs = np.linspace(0, len(sig)/20, len(sig))
#         plt.figure()
#         plt.plot(xs, sig)
##         for i in peaks:
##             plt.scatter(xs[i], sig[i], c='r')
#         for i in valley:
#             plt.scatter(xs[i], sig[i], c='dodgerblue')
#
#         for i in remove_peaks_to_close_to_valley:
#             plt.scatter(xs[i], sig[i], c='black')
#         for i in rm_oscillations:
#             plt.scatter(xs[i], sig[i], c='g')
#         for i in pb_val:
#             plt.scatter(xs[i], sig[i], c='r')
#         for i in pb_val2:
#             plt.scatter(xs[i], sig[i], c='c')
#         for i in new_peaks:
#             plt.scatter(xs[i], sig[i], c='y')

    return np.array(new_peaks[1:-1])


def detect_breathing_unwrap(sig, fs=20, window_peaks=4, amp_min=100):

    breathing_s = []
    amplitude_s = []
    for id_seg, seg in enumerate(sig):
        breathing, amplitude = detect_breathing(seg, window_peaks=window_peaks,
                                                amp_min=amp_min)
        breathing_s.append(breathing)
        amplitude_s.append(amplitude)

    return breathing_s, amplitude_s

def detect_breathing_unwrap2(sig, fs=20, window_peaks=4, amp_min=100):

    breathing_s = []

    for id_seg, seg in enumerate(sig):
        breathing = detect_breathing2(seg, window_peaks=window_peaks,
                                      amp_min=amp_min)
        breathing_s.append(breathing)

    return breathing_s


def detect_breathing_EDR(ecg, peaks, indicators, fs):
    """Detect breathings from ecg-derived respiration"""
    EDR_breathings = []
    if np.sum(indicators) == 0:
        EDR_breathings.append([])

    else:
        idx = np.where(np.array(indicators) == 1)[0]

        sep = np.where((idx[1:]-idx[:-1]) != 1)[0]
        if sep.tolist():
            ecg_i = ecg[idx[0:max(sep[0], 0)]]
            peaks_i = peaks[peaks > idx[0]]
            peaks_i = peaks_i[peaks_i < idx[max(sep[0], 0)]]
            peaks_i = np.array(peaks_i) - idx[0]
            if len(peaks_i) > 4:
                edr = extract_edr(ecg_i, peaks_i, fs)
                breathings = detect_peaks(edr, mph=np.median(edr), mpd=500)
                EDR_breathings.append(breathings)
            for j in range(1, len(sep)):
                id0 = idx[sep[j - 1] + 1]
                idmax = max(idx[sep[j]], 0)
                ecg_i = ecg[id0:idmax]
                peaks_i = peaks[peaks > id0]
                peaks_i = peaks_i[peaks_i < idmax]
                peaks_i = np.array(peaks_i) - id0

                if len(peaks_i) > 4:
                    edr = extract_edr(ecg_i, peaks_i, fs)
                    breathings = detect_peaks(edr, mph=np.median(edr), mpd=500)
                    EDR_breathings.append(breathings)
                else:
                    EDR_breathings.append([])

            id0 = idx[sep[-1]+1]
            ecg_i = ecg[id0:idx[-1]]
            peaks_i = peaks[peaks > id0]
            peaks_i = peaks_i[peaks_i < idx[-1]]
            peaks_i = np.array(peaks_i) - id0
            if len(peaks_i) > 4:
                edr = extract_edr(ecg_i, peaks_i, fs)
                breathings = detect_peaks(edr, mph=np.median(edr), mpd=500)
                EDR_breathings.append(breathings)
            else:
                EDR_breathings.append([])

        else:
            id0 = idx[0]
            idmax = idx[-1]
            ecg_i = ecg[id0:idx[-1]]
            peaks_i = peaks[peaks > id0]
            peaks_i = peaks_i[peaks_i < idmax]
            peaks_i = np.array(peaks_i) - id0
            if len(peaks_i) > 4:
                edr = extract_edr(ecg_i, peaks_i, fs)
                breathings = detect_peaks(edr, mph=np.median(edr), mpd=500)
                EDR_breathings.append(breathings)
            else:
                EDR_breathings.append([])


    rr = []
    for r in range(len(EDR_breathings)):
        seq = EDR_breathings[r]
        for points in range(1, len(seq)):
            rr.append((seq[points]-seq[points-1])/fs)
    return rr


def detect_breathing_EDR_unwrap(ecg, peaks, indicators, fs):

    rr = []
    for id_seg, seg in enumerate(peaks):
        EDR_breathings = detect_breathing_EDR(ecg[id_seg], peaks[id_seg],
                                              indicators[id_seg], fs)
        rr.append(EDR_breathings)
    return unwrap(rr)

def get_peaks_breath_amps(sig, peaks, fs):

    sig = np.array(sig)
    stat_amp = []
    if len(peaks) > 0:
        points_min = []
        points_max = []
        amp = []
        offset = int(1.5 * fs)
        for k in range(1, len(peaks)):
            if len(sig[peaks[k]-offset: peaks[k]+offset]):
                 max_ = peaks[k]-offset + np.argmax(sig[peaks[k]-offset:
                                                            peaks[k]+offset])
                 min_ = peaks[k]-offset + np.argmin(sig[peaks[k]-offset:
                                                            peaks[k]+offset])
                 points_min.append(min_)
                 points_max.append(max_)
                 amp.append(sig[max_]-sig[min_])
        stat_amp.extend(np.array(amp))
    return stat_amp

def detect_sleep(times, sig, fs, indicators_clean, 
                 ecg_fs, ecg_indicators_clean,
                 wait_time=1*60):

    sleep_time                  = None
    length                      = len(times)/fs
    if length < wait_time:
        length = wait_time
    checkpoint_number           = 5
    window_time                 = int(wait_time/checkpoint_number)
    window                      = window_time*fs
    threshold_var               = 40
    threshold_med               = 3000
    count_checkpoint            = 0
    count                       = 0
    for i in range(0, len(sig), window):
        count+=1
        imin        = i
        imax        = imin + window
        if imax >= len(sig):
            imax = len(sig) - 1
        seg             = sig[imin:imax]
        indicators_seg  = indicators_clean[imin:imax]
        median          = np.median(seg)
        iqr             = np.percentile(seg, 75) - np.percentile(seg, 25)

        ecg_imin            = int(imin*(ecg_fs/fs))
        ecg_imax            = int(imax*(ecg_fs/fs))
        ecg_indicators_seg  = ecg_indicators_clean[ecg_imin:ecg_imax]
        
        if iqr < threshold_var and median < threshold_med:
                sum_indicators      = sum(indicators_seg)
                ecg_sum_indicators  = sum(ecg_indicators_seg)
                if sum_indicators == 0 and ecg_sum_indicators < 25/100*len(ecg_indicators_seg):
                    count_checkpoint += 1
                    if count_checkpoint >= checkpoint_number:
                        sleep_time = times[imax]
                        break
        else:
            count_checkpoint = 0

    return sleep_time

def detect_wakeup(times, sigx, sigy, sigz, fs, wait_time=3):

    wakeup_time         = None

    length              = len(sigx)
    if length < wait_time:
        length = wait_time

    checkpoint_number   = 3
    window_time         = int(wait_time/checkpoint_number)
    window              = window_time*fs
    threshold_level     = 30
    times_levels        = []
    levels              = []
    count_checkpoint    = 0
    count               = 0
    for i in range(0, len(sigx), window):
        count+=1
        imin        = i
        imax        = imin + window
        if imax >= len(sigx):
            continue
        accx_seg        = sigx[imin:imax]
        accy_seg        = sigy[imin:imax]
        accz_seg        = sigz[imin:imax]
        sma             = signal_magnetude_area(accx_seg, accy_seg, accz_seg, fs)
        mean_activity   = np.mean(unwrap(sma))
        levels.append(mean_activity)
        times_levels.append(times[imax])
        if mean_activity > threshold_level:
            count_checkpoint += 1
            if count_checkpoint >= checkpoint_number:
                wakeup_time = times[imax]
                break

    return wakeup_time

def detect_worn(breath_1, accs, ecg):
    b_sig               = np.array(unwrap(breath_1.sig_))
    b_times             = np.array(unwrap(breath_1.times_))
    b_indicators_clean  = np.array(unwrap(breath_1.indicators_clean_2_))
    b_fs                = breath_1.fs_
    
    sigx                = np.array(unwrap(accs.accx.sig_filt_))
    sigy                = np.array(unwrap(accs.accy.sig_filt_))
    sigz                = np.array(unwrap(accs.accz.sig_filt_))
    a_times             = np.array(unwrap(accs.accx.times_))
    a_fs                = accs.accx.fs_
        
    ecg_indicators_clean = np.array(unwrap(ecg.indicators_clean_3_))
    ecg_fs              = ecg.fs_
    
    wakeup_time = None
    sleep_time = None
    if len(breath_1.times_) > 0 and len(accs.accx.times_) > 0:
        wakeup_time = min([accs.accx.times_[0][0], breath_1.times_[0][0]])
        sleep_time  = max([accs.accx.times_[-1][-1], breath_1.times_[-1][-1]])
        is_awake    = True
    
    sleep_times     = []
    still_times     = []
    move_times      = []
    wakeup_times    = []
    while wakeup_time is not None and sleep_time is not None:
        
        if is_awake:
            still_time          = None
            wait_time           = 1*60
            imin                = np.where(b_times >= wakeup_time)[0][0]
            b_seg               = b_sig[imin:]
            b_times_seg         = b_times[imin:]
            sleep_time          = detect_sleep(b_times_seg, b_seg, b_fs, b_indicators_clean, 
                                               ecg_fs, ecg_indicators_clean,
                                               wait_time=wait_time)
            if sleep_time is not None:
                is_awake        = False
                still_time      = sleep_time - np.timedelta64(int(wait_time), 's')
                sleep_times.append(sleep_time)
                still_times.append(still_time)
            
        if not is_awake:
            move_time           = None
            wait_time           = 6
            imin                = np.where(a_times >= sleep_time)[0]
            if len(imin) == 0:
                continue
            imin                = imin[0]
            segx                = sigx[imin:]
            segy                = sigy[imin:]
            segz                = sigz[imin:]
            a_times_seg         = a_times[imin:]
            wakeup_time         = detect_wakeup(a_times_seg, segx, segy, segz, a_fs, wait_time=wait_time)
            if wakeup_time is not None:
                is_awake        = True
                move_time       = wakeup_time - np.timedelta64(int(wait_time), 's')
                wakeup_times.append(wakeup_time)
                move_times.append(move_time)
        
        # if DEV:
            # if len(move_times) == 0 and len(still_times) > 0:
    if len(breath_1.times_) > 0 and len(accs.accx.times_) > 0:
        move_times.append(min([accs.accx.times_[-1][-1], breath_1.times_[-1][-1]]))
        wakeup_times.append(min([accs.accx.times_[-1][-1], breath_1.times_[-1][-1]]))
        
    return still_times, move_times


def detect_breathing_grenoble(sig, window_peaks=2.5, fs=32):
    """ Detect breath using detect_peaks function

    Parameters
    ----------
    sig: input respiratory signal
    fs


    Returns
    ----------
    peaks: breaths positions

#    """
    peaks = detect_peaks(sig, mpd=window_peaks*fs)
    valley = detect_peaks(sig, mpd=window_peaks*fs, valley=True)
    amp = get_peaks_breath_amps(sig, peaks, fs)

    diff = []
    id_peaks = -1
    for id_valley in valley:
       if peaks[peaks > id_valley].tolist():
           id_peaks = np.where(
               peaks == peaks[peaks > id_valley][0]
                               )[0][0]
           diff.append(abs(sig[peaks[id_peaks]]
                           - sig[id_valley]))
    diff = np.array(diff)
    remove_peaks_to_close_to_valley = []
    window_ = 4
    if peaks.tolist():
       for id_valley in valley:
           if peaks[peaks > id_valley].tolist():
               id_peaks = np.where(
                   peaks == peaks[peaks > id_valley][0]
                                   )[0][0]
           if id_peaks > -1:
               if id_peaks < window_:
                   range_amp = diff[: id_peaks + window_]
                   range_amp = np.median(range_amp[np.argsort(range_amp)[-window_:]])
               elif id_peaks > len(diff) - window_:
                   range_amp = diff[id_peaks - window_:]
                   range_amp = np.median(range_amp[np.argsort(range_amp)[-window_:]])
               else:
                   range_amp = diff[id_peaks-window_:
                                    id_peaks + window_]
                   range_amp = np.median(range_amp[np.argsort(range_amp)[-window_:]])
               if abs(sig[peaks[id_peaks]]
                      - sig[id_valley]) < 45*range_amp/100:
                   remove_peaks_to_close_to_valley.append(peaks[id_peaks])
    true_peaks = []
    for id_peak in peaks:
       if id_peak not in remove_peaks_to_close_to_valley:
           true_peaks.append(id_peak)
#    import matplotlib.pyplot as plt
#    if len(peaks) > 3:
#         xs = np.linspace(0, len(sig)/fs, len(sig))
#         plt.figure()
#         plt.plot(xs, sig)
##         for i in peaks2:
##             plt.scatter(xs[i], sig[i], c='r')
#         for i in valley:
#             plt.scatter(xs[i], sig[i], c='dodgerblue')
#         for i in true_peaks:
#             plt.scatter(xs[i], sig[i], c='g')

    return np.array(peaks[1:-1]), np.array(amp[1:-1]), np.array(true_peaks[1:-1])



def detect_qrs_grenoble(sample, sample_frequency=256, window=0.15):
    """ Detect qrs using detect_peaks function
        Input: sample, fe, window
        sample: input ecg
        sample_frequency
        window : window of integration
        Output: peaks
        peaks : QRS positions
    """

    window = int(window * sample_frequency)
    x_s = np.linspace(0, len(sample) / sample_frequency, len(sample))
    s_f = low_pass_filter(np.array(sample), 15, sample_frequency) - low_pass_filter(
        np.array(sample), 5, sample_frequency
    )
    x_s = np.linspace(0, len(s_f) / sample_frequency, len(s_f))
    der = (s_f[1:] - s_f[0:-1]) / ((x_s[1:] - x_s[0:-1]))
    a = der * der
    inte = []
    for k in range(len(a) - window):
        inte.append(integrate_trapz(a[k : k + window]))
    for k in range(len(a) - window, len(a)):
        inte.append(integrate_trapz(a[k:]))

    intef = np.array(inte)

    x_s = np.linspace(0, len(intef) / sample_frequency, len(intef))
    peaks = detect_peaks(
        intef,
        mph=0,
        mpd=int(0.25 * sample_frequency),
        threshold=0,
        edge=None,
        kpsh=False,
        valley=False
    )

    peak_values = intef[peaks]
    isort = np.argsort(peak_values)

    peaks_maxs_median = np.median(peak_values[isort[-10:]])

    keep_peaks = []
    for id_peak in range(len(peaks)):

        peak_value = intef[peaks[id_peak]]

        seg_peaks = intef[peaks[max(id_peak-5, 0):id_peak+5]]

        isort = np.argsort(seg_peaks)
        peaks_maxs_median = np.median(seg_peaks[isort[-3:]])

        if peak_value / peaks_maxs_median > 0.4:
            keep_peaks.append(peaks[id_peak])

    new_peaks = np.array(keep_peaks)

    return new_peaks

def detect_ecg_reversed(sig, fs, N=None):
    # IS REVERSE ECG ?
    IS_REVERSED_ECG = False
    sig             = np.array(sig) 
    mpd             = .5*fs
    if N is None:
        N           = len(sig)
    peaks           = detect_peaks(sig[:N], mpd=mpd)
    valleys         = detect_peaks(sig[:N], mpd=mpd, valley=True)
    
    sig_peaks       = sig[peaks]
    sig_valleys     = sig[valleys]
    if len(sig_peaks) > 3 and len(sig_valleys) > 3:
        if abs(np.percentile(sig_peaks, 90)) - abs(np.percentile(sig_valleys, 10)) < 0:
            IS_REVERSED_ECG = True
        
    return IS_REVERSED_ECG
