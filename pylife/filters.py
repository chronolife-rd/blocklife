import math
import numpy as np
from pylife.butterlife import Butterlife
import scipy.signal


def smooth(data_to_smooth, box_pts):
    """
    Smoothing
    Input: data_to_smooth, window
    Output: y_smooth
    y_smooth : smoothed data
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(data_to_smooth, box, mode='same')

    return y_smooth


def savitzky_golay(sig_to_filter, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only
                                                smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    if 2*len(sig_to_filter) < window_size:
        window_size = 2*len(sig_to_filter)+1
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    coeff_b = np.mat([[k**i for i in order_range] for k in range(-half_window,
                                                                 half_window+1)])
    coeff_m = np.linalg.pinv(coeff_b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = sig_to_filter[0] - np.abs(sig_to_filter[1:half_window+1][::-1]
                                          - sig_to_filter[0])
    lastvals = sig_to_filter[-1] + np.abs(sig_to_filter[-half_window-1:-1][::-1]
                                          - sig_to_filter[-1])
    sig_to_filter = np.concatenate((firstvals, sig_to_filter, lastvals))

    return np.convolve(coeff_m[::-1], sig_to_filter, mode='valid')


def filter_lms(noisy_signal, acc, coeff_mu, f_noisy_signal=200, fe_acc=50):
    """Least mean square
    Input: d, acc, mu, fe_d, fe_acc, show_plot
    noisy_signal : primary input (noisy signal to be filtered)
    acc : [accx, accy, accz] acceleration signal
    coeff_mu : learning rate
    f_noisy_signal : sampling rate primary input
    fe_acc : sampling rate acceleration
    Output:filtered signal
    """
    if((len(acc[:, 0]) != len(acc[:, 1])) or
       (len(acc[:, 0]) != len(acc[:, 2]))):
        print('Error : acceleration has not the same length on the 3 axes')
        return 0
    if f_noisy_signal != fe_acc:
        x_acc = np.linspace(0, len(acc)/fe_acc, len(acc))
        x_new = np.linspace(0, len(noisy_signal)/f_noisy_signal,
                            len(noisy_signal))
        acc = np.array([np.interp(x_new, x_acc, acc[:, 0]),
                        np.interp(x_new, x_acc, acc[:, 1]),
                        np.interp(x_new, x_acc, acc[:, 2])]).T
    noise_ = np.zeros((len(noisy_signal), 1))
    filtered_signal = np.zeros((len(noisy_signal), 1))
    coeff_matrix = np.array([np.random.random_sample((1, )),
                             np.random.random_sample((1, )),
                             np.random.random_sample((1, )),
                             np.random.random_sample((1, ))])
    for id_sig, sig in enumerate(noisy_signal):
        inter_vector = np.array([1, acc[id_sig, 0], acc[id_sig, 1],
                                 acc[id_sig, 2]]).reshape((1, 4))
        noise_ = np.dot(inter_vector, coeff_matrix)
        filtered_signal[id_sig] = sig - noise_
        coeff_matrix += (2*coeff_mu*filtered_signal[id_sig]*inter_vector).reshape((4, 1))

    return filtered_signal


def filter_LMS(d, x, mu):
    if not len(d) == len(x):
        raise ValueError('The length of vector d and matrix x must agree.')
    N = len(d)
    # Initialization
    n = 1
    y = np.zeros((N, n))
    e = np.zeros((N, n))
    w = np.random.random_sample((n,))
    # Adaptation loop
    # Computation of the estimate and the output
    for k in range(N):
        y[k] = w.T*x[k]
        e[k] = d[k] - y[k]
# Update of filter's weights
#            LMS
        dw = 2 * mu * e[k] * x[k]
#            Signed-Regressor LMS (SRLMS)
#            dw= 2* mu * e[k] * np.sign(x[k])
#            Sign LMS (SLMS)
#            dw= 2* mu * np.sign(e[k]) * x[k]
#            Sign-sign LMS (SSLMS)
#            dw= 2* mu * np.sign(e[k]) * np.sign(x[k])
        w += dw
    return y, e, w


def low_pass_filter(indata, cut_off=0.25, fs=50, padding=True):
    """
    Low pass filtering
    Input: indata, CutOff, Samplingrate
    indata : input signal
    CutOff: cut off frequency
    Samplingrate : Sampling frequency Hz
    Output: data
    data : filtered signal
    padding: Add a sample to keep signal's length
    """
    dat_2 = np.zeros((len(indata)+3, ))
    data = np.zeros((len(indata)-1, ))
    for i in range(len(indata)-1):
        dat_2[2 + i] = indata[i]
    dat_2[1] = dat_2[0] = indata[0]
    dat_2[len(indata)+2] = dat_2[len(indata)+1] = indata[len(indata)-1]

    w_c = math.tan(cut_off * math.pi / fs)
    coeff_a = (w_c * w_c) / (1 + (1.414213562 * w_c) + w_c * w_c)
    coeff_d = -2 * coeff_a + 2*coeff_a / (w_c * w_c)
    coeff_e = 1 - (2 * coeff_a) - 2*coeff_a / (w_c * w_c)

    dat_yt = np.zeros((len(indata)+3,))
    dat_yt[1] = dat_yt[0] = indata[0]
    for id_data in range(2, len(indata)+1):
        dat_yt[id_data] = coeff_a * dat_2[id_data]\
                    + 2*coeff_a * dat_2[id_data - 1]\
                    + coeff_a * dat_2[id_data - 2]\
                    + coeff_d * dat_yt[id_data - 1]\
                    + coeff_e * dat_yt[id_data - 2]
    dat_yt[len(indata)+2] = dat_yt[len(indata)+1] = dat_yt[len(indata)]
    dat_zt = np.zeros((len(indata)+1, ))
    dat_zt[len(indata)-1] = dat_yt[len(indata)+1]
    dat_zt[len(indata)] = dat_yt[len(indata)+2]
    for id_data in range(-(len(indata) - 1) + 1, 1):

        dat_zt[-id_data] = coeff_a * dat_yt[-id_data + 2]\
                    + 2 * coeff_a * dat_yt[-id_data + 3]\
                    + coeff_a * dat_yt[-id_data + 4]\
                    + coeff_d * dat_zt[-id_data + 1]\
                    + coeff_e * dat_zt[-id_data + 2]

    for id_data in range(len(indata) - 1):
        data[id_data] = dat_zt[id_data]

    if padding:
        padding_sample = data[0]
        dat = []
        dat.append(padding_sample)
        dat.extend(data)
        dat = np.array(dat)

    else:
        dat = data

    return dat


def low_high_pass_f(input_sig, f_sig, filter_order, w_c, type_f):
    """
    low_high_pass_f using butterworth filter
    Input: input_sig, f_sig, filter_order, w_c, type_f
    input_sig : signal to filter
    f_sig : signal sampling frequency
    filter_order : filter order
    w_c : cut off frequency
    type_f : "low" if low pass, "high" if high pass
    Output: output_sig
    """
    coeff_b, coeff_a = scipy.signal.butter(filter_order, w_c/(f_sig/2.),
                                           type_f)
    output_sig = scipy.signal.filtfilt(coeff_b, coeff_a, input_sig)
    return output_sig

# This filter is used !!!
def filter_breath_scipy(sig, fs):
    """ Filter breath signal

    Parameters
    ------------
    sig: breath signal
    fs: signal's sampling frequency

    Returns
    ------------
    Filtered signal

    """
    
    if len(sig) > fs:
        sig_filt = low_high_pass_f(sig, fs, 6, 1, 'low')
        sig_filt = low_high_pass_f(sig_filt, fs, 4, 0.12, 'high')
    else:
        sig_filt = sig - np.mean(sig) #[0]*len(sig) # sig # moyenne de signal (segment)

    return sig_filt

def filter_breath(sig, fs):
    """ Filter breath signal

    Parameters
    ------------
    sig: breath signal

    Returns
    ------------
    Filtered signal

    """
    # Old version
    # sig_filt = savitzky_golay(sig, 21, 2)

    # New version
    butter = Butterlife()
    
    sig_filt = butter.filt(sig, fs=fs, fc=0.8, order=4, ftype='low',
                           padding=True)
    
    # sig_filt = sig_filt - butter.filt(sig, fs=fs, fc=0.05, order=4,
                                      # ftype='low', padding=True)
    sig_filt = butter.filt(sig_filt, fs=fs, fc=0.12, order=4, ftype='high',
                           padding=True)

    return sig_filt


def filter_breath_unwrap(sig, fs):
    """ Filter breath signal

    Parameters
    ------------
    sig: breath signal

    Returns
    ------------
    Filtered signal

    """
    sig_filt_s = []
    for seg in sig:
        sig_filt = filter_breath_scipy(seg, fs)
        sig_filt_s.append(sig_filt)

    return sig_filt_s


def filter_ecg_scipy(sig, fs, low_order=10, high_order=4):
    """ Filter ECG signal

    Parameters
    ------------
    sig: ECG signal
    fs: signal's sampling frequency

    Returns
    ------------
    Filtered signal

    """
    # # Remove baseline
    # inter_1 = 0.465
    # inter_2 = 0.945
    # sig_med = filter_median(sig, kernel_size=2*int(inter_1*fs/2)+1)
    # sig_med = filter_median(sig_med, kernel_size=2*int(inter_2*fs/2)+1)
    # sig_med = sig - sig_med
    
    sig_filt = low_high_pass_f(sig, fs, low_order, 40, 'low')
    sig_filt = low_high_pass_f(sig_filt, fs, high_order, 1, 'high')

    return sig_filt


def filter_ecg_scipy_unwrap(sig, fs):
    """ Filter ECG signal

    Parameters
    ------------
    sig: ECG signal
    fs: signal's sampling frequency

    Returns
    ------------
    Filtered signal

    """
    sig_filt_s = []
    for seg in sig:
        sig_filt = filter_ecg_scipy(seg, fs)
        sig_filt_s.append(sig_filt)

    return sig_filt_s


def filter_ecg(sig, fs):
    # Remove baseline
    inter_1 = 0.465
    inter_2 = 0.945
    sig_med = filter_median(sig, kernel_size=2*int(inter_1*fs/2)+1)
    sig_med = filter_median(sig_med, kernel_size=2*int(inter_2*fs/2)+1)
    sig_med = sig - sig_med

    # Discret Filter
    butter = Butterlife()
    sig_low = butter.filt(sig_med, fs=fs, fc=40, order=4, ftype='low',
                          padding=True)
    sig_filt = butter.filt(sig_low, fs=fs, fc=0.5, order=4, ftype='high',
                           padding=True)

    return sig_filt


def filter_ecg_unwrap(sig, fs):
    """ Filter ECG signal

    Parameters
    ------------
    sig: ECG signal
    fs: signal's sampling frequency

    Returns
    ------------
    Filtered signal

    """
    sig_filt_s = []
    for seg in sig:
        sig_filt = filter_ecg(seg, fs)
        sig_filt_s.append(sig_filt)

    return sig_filt_s


def filter_acceleration(sig):
    """ Filter Acceleration signal

    Parameters
    ------------
    sig: Acceleration signal

    Returns
    ------------
    Filtered signal

    """
    sig_filt = sig - low_pass_filter(sig, padding=True)

    return sig_filt


def filter_acceleration_unwrap(sig):
    """ Filter Acceleration signal

    Parameters
    ------------
    sig: Acceleration signal

    Returns
    ------------
    Filtered signal

    """
    sig_filt_s = []
    for seg in sig:
        sig_filt = filter_acceleration(seg)
        sig_filt_s.append(sig_filt)

    return sig_filt_s


def filter_median(sig, kernel_size=3):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    sig = np.array(sig)
    assert kernel_size % 2 == 1, "Median filter kernel_size should be odd."
    assert sig.ndim == 1, "Input must be one-dimensional."

    k = (kernel_size - 1) // 2
    y = np.zeros((len(sig), kernel_size), dtype=sig.dtype)
    y[:, k] = sig
    for i in range(k):
        j = k - i
        y[j:, i] = sig[:-j]
        y[:j, i] = sig[0]
        y[:-j, -(i+1)] = sig[j:]
        y[-j:, -(i+1)] = sig[-1]

    sig_filt = np.median(y, axis=1)

    return sig_filt

#def smooth_rr(rr):
#
#    Q1 = np.percentile(rr, 25, interpolation = 'midpoint') 
#  
#    Q3 = np.percentile(rr, 75, interpolation = 'midpoint') 
#    
#    qd = (Q3 - Q1) / 2
#    
#    maximum_expected_diff = 3.32*qd
#    minimal_artifact_diff = np.median(rr) - 2.9*qd
#    
#    criterion = (maximum_expected_diff + minimal_artifact_diff)/2
#    
#    for i in range(len(rr)):
#         if rr[i] > criterion:
#              rr[i] = np.mean(rr)
#           
#    return rr
def smooth_rr(rr):

    mean_ = np.mean(rr)
    cut_off = 2 * np.std(rr)
    lower_limit = mean_ - cut_off
    upper_limit = mean_ + cut_off
        
    for i in range(len(rr)):
        if rr[i] <= lower_limit and rr[i] >= upper_limit:
            rr[i] = mean_
           
    return rr
