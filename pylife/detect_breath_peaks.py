# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.signal

def rsp_findpeaks(
    rsp_cleaned,
    sampling_rate,
    method="khodadad2018",
    amplitude_min=0.3,
    peak_distance=0.8,
    peak_prominence=0.5,
):
    """**Extract extrema in a respiration (RSP) signal**

    Low-level function used by :func:`.rsp_peaks` to identify inhalation and exhalation onsets
    (troughs and peaks respectively) in a preprocessed respiration signal using different sets of
    parameters. See :func:`.rsp_peaks` for details.

    Parameters
    ----------
    rsp_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.rsp_clean`.
    sampling_rate : int
        The sampling frequency of :func:`.rsp_cleaned` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"khodadad2018"`` (default), ``"scipy"`` or
        ``"biosppy"``.
    amplitude_min : float
        Only applies if method is ``"khodadad2018"``. Extrema that have a vertical distance smaller
        than(outlier_threshold * average vertical distance) to any direct neighbour are removed as
        false positive outliers. I.e., outlier_threshold should be a float with positive sign (the
        default is 0.3). Larger values of outlier_threshold correspond to more conservative
        thresholds (i.e., more extrema removed as outliers).
    peak_distance: float
        Only applies if method is ``"scipy"``. Minimal distance between peaks. Default is 0.8
        seconds.
    peak_prominence: float
        Only applies if method is ``"scipy"``. Minimal prominence between peaks. Default is 0.5.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which inhalation
        onsets and exhalation onsets occur, accessible with the keys ``"RSP_Troughs"`` and
        ``"RSP_Peaks"``, respectively.

    See Also
    --------
    rsp_clean, rsp_fixpeaks, rsp_peaks, signal_rate, rsp_amplitude, rsp_process, rsp_plot

    """
    # Try retrieving correct column
    if isinstance(rsp_cleaned, pd.DataFrame):
        try:
            rsp_cleaned = rsp_cleaned["RSP_Clean"]
        except NameError:
            try:
                rsp_cleaned = rsp_cleaned["RSP_Raw"]
            except NameError:
                rsp_cleaned = rsp_cleaned["RSP"]

    cleaned = np.array(rsp_cleaned)

    # Find peaks
    method = method.lower()  # remove capitalised letters
    if method in ["khodadad", "khodadad2018"]:
        info = _rsp_findpeaks_khodadad(cleaned, amplitude_min=amplitude_min)
    elif method == "biosppy":
        info = _rsp_findpeaks_biosppy(cleaned, sampling_rate=sampling_rate)
    elif method == "scipy":
        info = _rsp_findpeaks_scipy(
            cleaned,
            sampling_rate=sampling_rate,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
        )
    else:
        raise ValueError(
            "NeuroKit error: rsp_findpeaks(): 'method' should be one of 'khodadad2018', 'scipy' or 'biosppy'."
        )
    
    # Return info dictionary with peaks ans troughs 
    return info 

# =============================================================================
# Methods, we use khodadad
# =============================================================================
def _rsp_findpeaks_biosppy(rsp_cleaned, sampling_rate):
    """https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py"""

    extrema = _rsp_findpeaks_extrema(rsp_cleaned)
    extrema, amplitudes = _rsp_findpeaks_outliers(rsp_cleaned, extrema, amplitude_min=0)

    peaks, troughs = _rsp_findpeaks_sanitize(extrema, amplitudes)

    # Apply minimum period outlier-criterion (exclude inter-breath-intervals
    # that produce breathing rate larger than 35 breaths per minute.
    outlier_idcs = np.where((np.diff(peaks) / sampling_rate) < 1.7)[0]

    peaks = np.delete(peaks, outlier_idcs)
    troughs = np.delete(troughs, outlier_idcs)

    info = {"RSP_Peaks": peaks, "RSP_Troughs": troughs}
    return info


def _rsp_findpeaks_khodadad(rsp_cleaned, amplitude_min=0.3):
    """https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta"""
    peaks = []
    troughs = []
    
    extrema = _rsp_findpeaks_extrema(rsp_cleaned)
    extrema, amplitudes = _rsp_findpeaks_outliers(rsp_cleaned, extrema, amplitude_min=amplitude_min)
    if(len(extrema) >= 2):
        peaks, troughs = _rsp_findpeaks_sanitize(extrema, amplitudes)             

    info = {"RSP_Peaks": peaks, "RSP_Troughs": troughs}
    return info


def _rsp_findpeaks_scipy(rsp_cleaned, sampling_rate, peak_distance=0.8, peak_prominence=0.5):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html"""
    peak_distance = sampling_rate * peak_distance
    peaks, _ = scipy.signal.find_peaks(
        rsp_cleaned, distance=peak_distance, prominence=peak_prominence
    )
    troughs, _ = scipy.signal.find_peaks(
        -rsp_cleaned, distance=peak_distance, prominence=peak_prominence
    )

    info = {"RSP_Peaks": peaks, "RSP_Troughs": troughs}
    return info


# =============================================================================
# Internals
# =============================================================================

def _rsp_findpeaks_extrema(rsp_cleaned):
    # Detect zero crossings (note that these are zero crossings in the raw
    # signal, not in its gradient).
    greater = rsp_cleaned > 0
    smaller = rsp_cleaned < 0
    risex = np.where(np.bitwise_and(smaller[:-1], greater[1:]))[0]
    fallx = np.where(np.bitwise_and(greater[:-1], smaller[1:]))[0]

    allx = np.concatenate((risex, fallx))
    allx.sort(kind="mergesort")
    
    # Add first and last extrema of the signal 
    first_x = 0
    last_x = len(rsp_cleaned) - 1
    allx = np.insert(allx, 0, first_x)
    allx = np.append(allx, last_x)
    allx = np.unique(allx)

    # Find extrema by searching minima between falling zero crossing and
    # rising zero crossing, and searching maxima between rising zero
    # crossing and falling zero crossing.
    extrema = []

    for i in range(len(allx) - 1):
        beg = allx[i]
        end = allx[i + 1]
        
        min_extreme = np.argmin(rsp_cleaned[beg:end])
        max_extreme = np.argmax(rsp_cleaned[beg:end])
        min_value = abs(rsp_cleaned[beg + min_extreme])
        max_value = abs(rsp_cleaned[beg + max_extreme])
        
        if(min_value >=  max_value):
            extreme_to_add = min_extreme
        else: extreme_to_add = max_extreme
    
        extrema.append(beg + extreme_to_add)
    
    extrema = np.asarray(extrema)
    
    # Remove extrema if == x crossing at beggining and at the end of signal
    if (extrema[0] == allx[0]):
        extrema = extrema[1:]
    if (extrema[-1] == allx[-1]):
        extrema = extrema[:-1]
        
    # # View result
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.title("Extremas found by searching minima and maxima between 2 zero crossing ")
    # plt.plot(rsp_cleaned, color = 'black', label = 'sig')
    # plt.scatter(allx, rsp_cleaned[allx], color = 'b', label = 'allx', s = 100, marker = 'x')
    # plt.scatter(extrema, rsp_cleaned[extrema], color = 'blue', label = 'All extremas')
    # plt.legend()
    
    return extrema

def _rsp_findpeaks_outliers(rsp_cleaned, extrema, amplitude_min=0.3):

    # Only consider those extrema that have a minimum vertical distance to
    # their direct neighbor, i.e., define outliers in absolute amplitude
    # difference between neighboring extrema.
    vertical_diff = np.abs(np.diff(rsp_cleaned[extrema]))
    median_diff = np.median(vertical_diff)
    min_diff = np.where(vertical_diff > (median_diff * amplitude_min))[0]
    
    # Clean new extrema and add the last extrema
    extrema_2 = extrema[min_diff]
    extrema_2 = np.append(extrema_2, extrema[-1])

    # Make sure that the alternation of peaks and troughs is unbroken. If
    # alternation of sign in extdiffs is broken, remove the extrema that
    # cause the breaks.
    amplitudes_2 = rsp_cleaned[extrema_2]
    extdiffs = np.sign(np.diff(amplitudes_2))
    extdiffs = np.add(extdiffs[0:-1], extdiffs[1:])
    removeext = np.where(extdiffs != 0)[0] + 1
    
    # Clean extrema and amplitudes
    extrema_3 = np.delete(extrema_2, removeext)
    amplitudes_3 = np.delete(amplitudes_2, removeext)
    
    return extrema_3, amplitudes_3


def _rsp_findpeaks_sanitize(extrema, amplitudes):
    # To be able to consistently calculate breathing amplitude, make sure that
    # the extrema always start with a trough and end with a peak, since
    # breathing amplitude will be defined as vertical distance between each
    # peak and the preceding trough. Note that this also ensures that the
    # number of peaks and troughs is equal.
    
    if (amplitudes[0] > amplitudes[1] and amplitudes[-1] > amplitudes[-2]):
        peaks = extrema[0::2]
        troughs = extrema[1:-1:2]
        
    if (amplitudes[0] > amplitudes[1] and amplitudes[-1] < amplitudes[-2]):
        peaks = extrema[0:-1:2]
        troughs = extrema[1::2]
        
    if (amplitudes[0] < amplitudes[1] and amplitudes[-1] > amplitudes[-2]):
        peaks = extrema[1::2]
        troughs = extrema[0:-1:2]
        
    if (amplitudes[0] < amplitudes[1] and amplitudes[-1] < amplitudes[-2]):
        peaks = extrema[1:-1:2]
        troughs = extrema[0::2]


    return peaks, troughs
