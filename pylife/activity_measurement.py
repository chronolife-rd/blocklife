# --- Add imports for PROD and DEV env
import math
import numpy as np
from pylife.env import get_env
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import matplotlib.pyplot as plt


def compute_activity_level(acc_x, acc_y, acc_z):
    """Activity level computation
    Input : acc_x, acc_y, acc_z, n
    acc_x : acc_x component of the acceleration signal
    acc_y : y component of the acceleration signal
    acc_z : z component of the acceleration signal
   
    Output :
    activity level
    """
    if not len(acc_x) == len(acc_y) or not len(acc_y) == len(acc_z):
        raise ValueError("The length of vectors X, Y and Z must agree.")
   
    activity_level = (abs(acc_x) + abs(acc_y) + abs(acc_z))/3

    activity_level = activity_level  

    return activity_level


def compute_activity_level_unwrap(acc_x, acc_y, acc_z):
    """Activity level computation
    """
    activity_level_ = []

    for id_seg, seg in enumerate(acc_x):
        seg_x = acc_x[id_seg]
        seg_y = acc_y[id_seg]
        seg_z = acc_z[id_seg]
        activity_level = compute_activity_level(seg_x, seg_y, seg_z)
        activity_level_.append(activity_level)

    return activity_level_


def detect_peaks_steps(x_sig, mph=None, mpd=1, threshold=0, edge="rising",
                       kpsh=False, valley=False):
    """Detect peaks in data based on their amplitude and other features.
    """

    x_sig = np.atleast_1d(x_sig).astype("float64")
    if x_sig.size < 3:
        return np.array([], dtype=int)
    if valley:
        x_sig = -x_sig
    # find indices of all peaks
    d_x = x_sig[1:] - x_sig[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x_sig))[0]
    if indnan.size:
        x_sig[indnan] = np.inf
        d_x[np.where(np.isnan(d_x))[0]] = np.inf
    i_ne, i_re, i_fe = np.array([[], [], []], dtype=int)
    if not edge:
        i_ne = np.where((np.hstack((d_x, 0)) < 0)
                        & (np.hstack((0, d_x)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            i_re = np.where((np.hstack((d_x, 0)) <= 0)
                            & (np.hstack((0, d_x)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            i_fe = np.where((np.hstack((d_x, 0)) < 0)
                            & (np.hstack((0, d_x)) >= 0))[0]
    ind = np.unique(np.hstack((i_ne, i_re, i_fe)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))),
                invert=True
            )
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x_sig.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x_sig[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        d_x = np.min(np.vstack([x_sig[ind] - x_sig[ind - 1],
                                x_sig[ind] - x_sig[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(d_x < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x_sig[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x_sig[ind[i]] > x_sig[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def signal_magnetude_area(acc_x, acc_y, acc_z, n_window):
    """Signal Magnitude Area computation
    Input : acc_x, acc_y, acc_z, n
    acc_x : acc_x component of the acceleration signal
    acc_y : y component of the acceleration signal
    acc_z : z component of the acceleration signal
    n_window : size of the window used for SMA computation in number of samples
    Output :
    sma: acceleration magnitude summations over three axes
    of each window normalized by the window length
    """
    if not len(acc_x) == len(acc_y) or not len(acc_y) == len(acc_z):
        raise ValueError("The length of vectors X, Y and Z must agree.")
    sma = np.zeros((len(acc_x), 1))
    if len(sma) > n_window:
         for i in range(n_window):
             l_norm = i + n_window
             sma[i] = (1 / (l_norm)) * (
                 np.sum(abs(acc_x[0: i + n_window]))
                 + np.sum(abs(acc_y[0: i + n_window]))
                 + np.sum(abs(acc_z[0: i + n_window]))
             )
     
         for i in range(len(acc_x) - n_window):
             sma[i] = (1 / (2 * n_window)) * (
                 np.sum(abs(acc_x[i - n_window: i + n_window]))
                 + np.sum(abs(acc_y[i - n_window: i + n_window]))
                 + np.sum(abs(acc_z[i - n_window: i + n_window]))
             )
     
         for i in range(len(acc_x) - n_window, len(acc_x)):
             l_norm = len(acc_x) - (i - n_window)
             sma[i] = (1 / (l_norm)) * (
                 np.sum(abs(acc_x[i - n_window:]))
                 + np.sum(abs(acc_y[i - n_window:]))
                 + np.sum(abs(acc_z[i - n_window:]))
             )

    return sma

def signal_magnetude_area_unwrap(acc_x, acc_y, acc_z, n_window):
    """Signal Magnitude Area computation
    """
    sma_ = []

    for id_seg, seg in enumerate(acc_x):
        seg_x = acc_x[id_seg]
        seg_y = acc_y[id_seg]
        seg_z = acc_z[id_seg]
        sma = signal_magnetude_area(seg_x, seg_y, seg_z, n_window)
        sma_.append(sma)

    return sma_


def low_pass_filter_activity(indata, cut_off=0.25, sampling_rate=50):
    """
    Low pass filtering
    Input: indata, CutOff, Samplingrate
    indata : input signal
    CutOff: cut off frequency
    Samplingrate : Sampling frequency Hz
    Output: data
    data : filtered signal
    """
    dat_2 = np.zeros((len(indata) + 3, ))
    data = np.zeros((len(indata) - 1, ))
    for i in range(len(indata) - 1):
        dat_2[2 + i] = indata[i]
    dat_2[1] = dat_2[0] = indata[0]
    dat_2[len(indata) + 2] = dat_2[len(indata) + 1] = indata[len(indata) - 1]

    w_c = math.tan(cut_off * math.pi / sampling_rate)
    coeff_a = (w_c * w_c) / (1 + (1.414213562 * w_c) + w_c * w_c)
    coeff_d = - 2 * coeff_a + 2 * coeff_a / (w_c * w_c)
    coeff_e = 1 - (2 * coeff_a) - 2 * coeff_a / (w_c * w_c)
    dat_yt = np.zeros((len(indata) + 3,))
    dat_yt[1] = dat_yt[0] = indata[0]
    for id_data in range(2, len(indata) + 1):
        dat_yt[id_data] = coeff_a * dat_2[id_data]\
            + 2 * coeff_a * dat_2[id_data - 1]\
            + coeff_a * dat_2[id_data - 2]\
            + coeff_d * dat_yt[id_data - 1]\
            + coeff_e * dat_yt[id_data - 2]
    dat_yt[len(indata) + 2] = dat_yt[len(indata) + 1] = dat_yt[len(indata)]
    dat_zt = np.zeros((len(indata) + 1, ))
    dat_zt[len(indata) - 1] = dat_yt[len(indata) + 1]
    dat_zt[len(indata)] = dat_yt[len(indata) + 2]
    for id_data in range(- (len(indata) - 1) + 1, 1):

        dat_zt[- id_data] = coeff_a * dat_yt[- id_data + 2]\
            + 2 * coeff_a * dat_yt[- id_data + 3]\
            + coeff_a * dat_yt[- id_data + 4]\
            + coeff_d * dat_zt[- id_data + 1]\
            + coeff_e * dat_zt[- id_data + 2]
    for id_data in range(len(indata) - 1):
        data[id_data] = dat_zt[id_data]
    padding = data[0]
    dat = []
    dat.append(padding)
    dat.extend(data)
    return np.array(dat)


def step_counting(data, sma_window=50, sma_threshold=15, samp_rate=50):
    """ Step counting
            Input: data, sma_window, sma_threshold
            data : Nx3 matrix, with N number of samples[acc_x, acc_y, acc_z]
            sma_window : size of the window used for the sma calculation
            sma_threshold : walk detection theshold
            Output: Number of steps
    """
    facc_mag = data[:, 0] - low_pass_filter_activity(data[:, 0], cut_off=0.25,
                                            sampling_rate=samp_rate)
    facc_mag = low_pass_filter_activity(facc_mag, cut_off=2,
                                        sampling_rate=samp_rate)
    sma = signal_magnetude_area(facc_mag, facc_mag, facc_mag, sma_window)
    idx = np.where(np.array(sma) > sma_threshold)[0]
    segment = idx[1:] - idx[0:-1]
    split = np.where(segment != 1)[0]
    seg = []
    raw_seg = []
    if len(split) == 0 and len(idx) != 0:
        seg.append(facc_mag[idx])
        raw_seg.append(data[:, 0][idx])
    elif len(split) == 1:
        seg.append(facc_mag[idx[0]:idx[split[0]]])
        seg.append(facc_mag[idx[split[0] + 1]:idx[-1]])

        raw_seg.append(data[:, 0][idx[0]: idx[split[0]]])
        raw_seg.append(data[:, 0][idx[split[0] + 1]: idx[- 1]])
    else:
        for j in range(len(split)):
            if j == 0:
                seg.append(facc_mag[idx[0]:idx[split[j]]])
                seg.append(facc_mag[idx[split[j] + 1]:idx[split[j + 1]]])

                raw_seg.append(data[:, 0][idx[0]: idx[split[j]]])
                raw_seg.append(data[:, 0][idx[split[j] + 1]:
                                          idx[split[j + 1]]])
            elif j == len(split) - 1:
                seg.append(facc_mag[idx[split[j] + 1]: idx[-1]])
                raw_seg.append(data[:, 0][idx[split[j] + 1]: idx[-1]])

            else:
                seg.append(facc_mag[idx[split[j] + 1]: idx[split[j + 1]]])
                raw_seg.append(data[:, 0][idx[split[j] + 1]: idx[split[j + 1]]])
    count = 0
    for id_seg in range(len(seg)):
        if len(seg[id_seg]) > 0:
            mo_amp = np.mean([math.sqrt(x * x + x * x
                                        + x * x) for x in seg[id_seg]])
            if mo_amp <= 20:
                mpd = 28
                threshold_peak = 5
            if 20 < mo_amp <= 30:
                mpd = 22
                threshold_peak = 7
            elif 30 < mo_amp <= 40:
                mpd = 20
                threshold_peak = 12
            elif 40 < mo_amp <= 60:
                mpd = 20
                threshold_peak = 18
            elif 60 < mo_amp <= 90:
                mpd = 10
                threshold_peak = 20
            elif 90 < mo_amp:
                mpd = 2
                threshold_peak = -1000
            pks_ = detect_peaks_steps(seg[id_seg], mph=threshold_peak,
                                      mpd=mpd)
            keep_peaks = []
            for i in pks_:
                offset_min = np.max([0, i - int(mpd)])
                offset_max = np.min([len(raw_seg[id_seg]), i + int(mpd)])
                neg_value = len(np.where(raw_seg[id_seg][offset_min:
                                                         offset_max] < 1900)[0])
                if neg_value > 0.15 * len(raw_seg[id_seg][offset_min:offset_max]):
                    keep_peaks.append(i)
            count += len(keep_peaks)
    return count


def compute_n_steps(times, acc_x, acc_y, acc_z, fs):
    """ Compute number of steps

        Parameters
        ---------------------------------
        times: Signal times
        acc x, y, z: Acceleration signals
        fs: sampling frequency

        Returns
        ---------------------------------
        n_steps: Number of steps
        n_steps_times_start: Times start for each number of step calculation
        n_steps_times_stop: Times stop for each number of step calculation

    """
    n_steps = []
    steps = []
    steps_times_start = []
    steps_times_stop = []

    sig = np.vstack((acc_x, acc_y))
    sig = np.vstack((sig, acc_z))
    sig = sig.T
    list_steps = []
    count = 0
    for j in range(int(len(sig)/(fs*10))):
        imin = (fs*10)*j
        imax = (fs*10)*(j+1)

        
        if j == int(len(sig)/(fs*10))-1:
            imax = -1
        times_dat_start = times[imin]
        times_dat_stop = times[imax]
        dat = sig[imin:imax]
        c = step_counting(dat)
        list_steps.append([times[imin], c])
        count += c
        n_steps.append(count)
        steps.append(c)
        steps_times_start.append(times_dat_start)
        steps_times_stop.append(times_dat_stop)

    return n_steps, steps, steps_times_start, steps_times_stop


def compute_n_steps_unwrap(times, acc_x, acc_y, acc_z, fs):
    """ Compute number of steps

        Parameters
        ---------------------------------
        times: Signal times
        acc x, y, z: Acceleration signals
        fs: sampling frequency

        Returns
        ---------------------------------
        n_steps: Number of steps
        n_steps_times_start: Times start for each number of step calculation
        n_steps_times_stop: Times stop for each number of step calculation

    """
    n_steps_s = []
    steps_s = []
    steps_times_start_s = []
    steps_times_stop_s = []

    for id_seg, seg in enumerate(acc_x):
        steps = []
        n_steps = []
        steps_times_start = []
        steps_times_stop = []
        
        if len(seg) > 50:
            times_seg = times[id_seg]
            seg_x = acc_x[id_seg]
            seg_y = acc_y[id_seg]
            seg_z = acc_z[id_seg]

            n_steps, steps, steps_times_start,\
                steps_times_stop = compute_n_steps(times_seg, seg_x,
                                                     seg_y, seg_z, fs)
        n_steps_s.append(n_steps)
        steps_s.append(steps)
        steps_times_start_s.append(steps_times_start)
        steps_times_stop_s.append(steps_times_stop)
        
    return n_steps_s, steps_s, steps_times_start_s, steps_times_stop_s