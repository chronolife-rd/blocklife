# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:45:18 2021

@author: RamziAbdelhafidh
"""


import sys
path_pylife = 'C:/Users/Blandinelandrieu/Desktop/Chronolife_local_sandbox/pylife'  #' ... '
sys.path.append(path_pylife)
from pylife.env import get_env

DEV = get_env()
import numpy as np
from pylife.detection import detect_qrs, detect_peaks_r


#import matplotlib.pyplot as plt

def compute_TA(T: np.ndarray, w: int) -> np.ndarray:
    """
    A function that computes TA amplitude for a given template of a signal
    :param T: template of signal to compute TA on
    :type T: np.ndarray
    :param w: window length
    :type w: int

    """
    TA = []
    for i in range(T.shape[0]):
        if i+w < T.shape[0]:
            seg = T[i:i+w]
            TA.append(max(seg)-min(seg))
        else:
            TA.append(T[i])
    return np.array(TA)


def compute_TDA(T: np.ndarray, w: int) -> np.ndarray:
    """
    A function that computes TDA amplitude for a given template of a signal
    :param T: template of the signal to compute TDA on
    :type T: np.ndarray
    :param w: window
    :type w: int

    """
    # compute derivative
    deriv = np.array([0] + [T[i]-T[i-1] for i in range(1, T.shape[0])])
    
    # compute TDA based on derivative
    TDA = []
    for i in range(deriv.shape[0]):
        if i+w < T.shape[0]:
            seg = deriv[i:i+w]
            TDA.append(max(seg)-min(seg))
        else:
            TDA.append(deriv[i])
    return np.array(TDA)


def compute_qrs_offset(T: np.ndarray, TA: np.ndarray, TDA: np.ndarray,
                       TT: float, TD: float, r_instant: int) -> int:
    """
    A function that computes the offset of the QRS complex
    :param T: template of 200ms that follows the R pic
    :type T: np.ndarray
    :param TA: TA amplitude of the template T
    :type TA: np.ndarray
    :param TDA: TDA amplitude of the template T
    :type TDA: np.ndarray
    :param TT: threshold for the TA amplitude to consider while finding 
    the QRS offset
    :type TT: float
    :param TD: threshold for the TDA amplitude to consider while finding
    the QRS offset
    :type TD: float
    :param r_instant: R pic sample index in the whole PQRS complex segment
    :type r_instant: int

    """
    cond = False
    i = 0
    offset = r_instant
    while (cond == False) and (i < T.shape[0]):
        if (TA[i] < TT) or (TDA[i] < TD):
            cond = True
            offset = i + r_instant
        else:
            i += 1
    return offset


def to_segments(signal: np.ndarray,
                qrs_instants: np.ndarray,
                fs: int) -> list[dict]:
    """
    A function that divides an ECG signal to PQRST segments based on QRS onset
    detected instants
    :param signal: signal to extract PQRST segments on
    :type signal: np.ndarray
    :param qrs_instants: QRS onset positions in the signal
    :type qrs_instants: np.ndarray
    :param fs: sampling frequency
    :type fs: int
    :return: list of templates
    :rtype: list[dict]

    """
    pqrst_onset = int(0.150 * fs)  # 150ms
    templates = []
    for i in range(1, qrs_instants.shape[0]-1):
        # template debut position in the whole signal
        template_debut_instant = qrs_instants[i] - pqrst_onset
        # template finish position in the whole signal
        template_finish_instant = qrs_instants[i+1] - int(0.200 * fs)
        # extract template based on debut and finish instants
        template_signal = signal[template_debut_instant:template_finish_instant]
        templates.append({'signal': template_signal,
                          'template_instant': template_debut_instant})
    return templates


def add_r_peaks_to_templates(templates: list,
                             r_peaks_instants: np.ndarray,
                             qrs_instants: np.ndarray,
                             fs: int) -> list[dict]:
    """
    A function that adds R pics relative instants to templates
    :param templates: list of PQRST templates to add relative R peaks 
    instants to
    :type templates: list
    :param r_peaks_instants: R peaks absolute instants
    :type r_peaks_instants: np.ndarray
    :param qrs_instants: QRS onset absolute instants
    :type qrs_instants: np.ndarray
    :param fs: sampling frequency
    :type fs: int

    """
    pqrst_onset = int(0.150 * fs)  # 150ms
    for i in range(1, len(r_peaks_instants)-1):
        r_instant = r_peaks_instants[i] - (qrs_instants[i]-pqrst_onset)
        templates[i-1]['r_instant'] = r_instant

    return templates


def add_qrs_onset_to_templates(templates: list,
                               qrs_instants: np.ndarray,
                               r_peaks_instants: np.ndarray) -> list[dict]:
    """
    A function that adds QRS onset relative instants to templates
    :param templates: list of PQRST templates to add relative QRS onset 
    instants to
    :type templates: list
    :param qrs_instants: absolute QRS onset instants
    :type qrs_instants: np.ndarray
    :param r_peaks_instants: absolute R peaks instants
    :type r_peaks_instants: np.ndarray
    :return: templates to add QRS onset relative instants to
    
    """
    for i in range(1, len(r_peaks_instants)-1):
        qrs_instant_template = templates[i-1]['r_instant'] - \
            (r_peaks_instants[i]-qrs_instants[i])
        templates[i-1]['qrs_onset'] = qrs_instant_template
    return templates


def add_qrs_offset_to_templates(templates: list, w: int,
                                c1: float, c2: float, fs: int) -> list[dict]:
    """
    A function that adds QRS offset instants to templates 
    :param templates: PQRST templates to add QRS offset instants to
    :type templates: list
    :param w: window parameter for TA and TDA computation 
    :type w: int
    :param c1: parameter for TT computation
    :type c1: float
    :param c2: parameter for TD computation
    :type c2: float
    :param fs: sampling frequency
    :type fs: int

    """
    for template in templates:
        signal = template['signal']
        #### modif a verifier
        r_instant = template['r_instant']
        if len(signal)>(r_instant+1):
            T = signal[r_instant:(r_instant+int(0.200*fs))]
            TA = compute_TA(T, w)
            TDA = compute_TDA(T, w)
            TT = c1 * (max(TA)-min(TA)) + min(TA)
            TD = c2 * (max(TDA)-min(TDA)) + min(TDA)
            offset = compute_qrs_offset(T, TA, TDA, TT, TD, r_instant)
        else :
            offset = 0
       
        template['qrs_offset'] = offset
    return templates


# def t_wave_is_reversed(t_wave):
#     centered_t_wave = np.array(t_wave) - np.mean(t_wave)
#     absolute_centered_t_wave = np.abs(centered_t_wave)
#     if centered_t_wave[np.argmax(absolute_centered_t_wave)] > 0:
#         return False
#     else:
#         return True


def add_T_peaks_to_templates(templates: list, fs: int) -> list:
    """
    A function that detectes T peak of a PQTST template
    :param templates: PQRST template
    :type templates: list
    :param fs: sampling frequency
    :type fs: int
    :return: templates with detected T peaks relative instants
    :rtype: list

    """
    for template in templates:
        signal = template['signal']
        qrs_offset = template['qrs_offset']
        #### change for signal[qrs_offset :] empty
        #print('len(signal),qrs_offset',len(signal), qrs_offset )
        T = signal[qrs_offset-1 :]
        if len(T)<1:
            T_peak = 0
        else :
            T_peak = np.argmax(T)
        template['T_peak'] = T_peak + qrs_offset
    return templates


def add_T_offset_to_templates(templates: list) -> list:
    """
    A function that computes T offset instant in a PQRST template
    :param templates: PQRST templates to add T offset relative instants to
    :type templates: list
    :return: PQRST templates with added T offset relative instants
    :rtype: list

    """
    for template in templates:
        signal = template['signal']
        T_peak = template['T_peak']
        # consider a template from the T peak instant until the end of 
        #PQRST template
        T = signal[T_peak:]
        if len(T) >= 2:
            # compute the gradient of the g line
            k = (T[-1] - T[0])/(len(T)-1-0)
            # y-intercept of the g straight line
            d = T[0]
            # g straight line
            g = [k*i+d for i in range(len(T))]
            # S = ECG - g
            S = np.array(T) - np.array(g)
            T_offset = np.argmin(S) + T_peak
        else:
            T_offset = T_peak
        template['T_offset'] = T_offset
    return templates

def detect_peaks_Q_unwrap(sig, fs, peaks_qrs):
    peaks_r_ = []
    amplitude_ = []
    for i, seg in enumerate(sig):
        peaks_qrs_seg = peaks_qrs[i]
        peaks_r, amp = detect_peaks_r(seg, fs, peaks_qrs_seg)
        peaks_r_.append(peaks_r)
        amplitude_.append(amp)

    return peaks_r_, amplitude_

def detect_peaks_Q(sig, fs, peaks_R):
    """
    Get Q peak by searching local minima in the 0.8 s before R peak
    peaks_R are indexes
    peaks_q are indexes
    
    """
    sig         = np.array(sig)
    peaks_Q    = []

    dec = int(0.08*fs)
    for peak in peaks_R:
        seg         = sig[max(peak-dec, 0):peak]
        imin = np.argmin(seg)
        peak = peak - imin
        peaks_Q.append(peak)
        
    return np.array(peaks_Q)

def detect_QT_interval(signal: np.ndarray, fs: int, peaks_R:np.ndarray) -> tuple:
    
    """
    peaks_R : indexes
    
    """
    
    if len(peaks_R)<1:
        return [], []
    
    q_wave = detect_peaks_Q(sig = signal,
                                  fs = fs, 
                                  peaks_R = peaks_R)
    
    # Segmentation
    templates = to_segments(signal=signal,
                            qrs_instants = q_wave,
                            fs=fs)
    # Add R peaks to templates
    templates = add_r_peaks_to_templates(
        templates, peaks_R, q_wave, fs)
    
    # Add QRS onsets to templates
    templates = add_qrs_onset_to_templates(
        templates, q_wave, peaks_R)
    
    # Add QRS offstes to templates
    w_qrs_offset = int(0.060 * fs)  # 60ms
    c1 = 0.02  # 2%
    c2 = 0.02  # 2%
    
    #print('templates', templates)
    
    templates = add_qrs_offset_to_templates(templates,
                                            w_qrs_offset,
                                            c1,
                                            c2,
                                            fs)
    # Add T peaks to templates
    templates = add_T_peaks_to_templates(templates, fs)
    # Add T offset to templates
    templates = add_T_offset_to_templates(templates)

    res = [{info: template[info] for info in ['signal',
                                              'template_instant',
                                              'qrs_onset',
                                              'T_offset']} for template in templates]

    detected_qrs_onsets = [res[i]['qrs_onset'] + res[i]
                           ['template_instant'] for i in range(len(res))]

    detected_t_offsets = [res[i]['T_offset'] + res[i]
                          ['template_instant'] for i in range(len(res))]

    return detected_qrs_onsets, detected_t_offsets



# def detect_QT_interval(signal: np.ndarray, fs: int) -> tuple:
#     qrs_instants = detect_qrs(sig=signal,
#                               fs=fs,
#                               window=0.15)
#     r_peaks_instants, r_peaks_amplitudes = detect_peaks_r(sig=list(signal),
#                                                           fs=fs,
#                                                           peaks=qrs_instants)
#     # Segmentation
#     templates = to_segments(signal=signal,
#                             qrs_instants=qrs_instants,
#                             fs=fs)
#     # Add R peaks to templates
#     templates = add_r_peaks_to_templates(
#         templates, r_peaks_instants, qrs_instants, fs)
#     # Add QRS onsets to templates
#     templates = add_qrs_onset_to_templates(
#         templates, qrs_instants, r_peaks_instants)
#     # Add QRS offstes to templates
#     w_qrs_offset = int(0.060 * fs)  # 60ms
#     c1 = 0.02  # 2%
#     c2 = 0.02  # 2%
#     templates = add_qrs_offset_to_templates(templates,
#                                             w_qrs_offset,
#                                             c1,
#                                             c2,
#                                             fs)
#     # Add T peaks to templates
#     templates = add_T_peaks_to_templates(templates, fs)
#     # Add T offset to templates
#     templates = add_T_offset_to_templates(templates)

#     res = [{info: template[info] for info in ['signal',
#                                               'template_instant',
#                                               'qrs_onset',
#                                               'T_offset']} for template in templates]

#     detected_qrs_onsets = [res[i]['qrs_onset'] + res[i]
#                            ['template_instant'] for i in range(len(res))]

#     detected_t_offsets = [res[i]['T_offset'] + res[i]
#                           ['template_instant'] for i in range(len(res))]

#     return detected_qrs_onsets, detected_t_offsets
