import numpy as np
from pylife.remove import remove_disconnection_multi, remove_peaks,\
    remove_peaks_unwrap
from pylife.detection import detect_breathing_unwrap2,\
    detect_breathing2, detect_qrs_brassiere_unwrap, detect_breathing_EDR_unwrap,\
    detect_breathing_EDR, detect_qrs_unwrap, detect_qrs, detect_qrs_brassiere
from pylife.useful import is_list_of_list, unwrap

def compute_deltas_breath(timeline, sig, indicators_, nb_delta, delta, amp,\
                          fs):

    t = timeline[0]
    rr_delta = []
    bpm_delta = []
    hrv_delta = []
    times_delta = []
    while t < timeline[-1]:
        t_delta = timeline[timeline > t]
        resp_delta = sig[timeline > t]
        ind_delta = indicators_[timeline > t]

        resp_delta = resp_delta[t_delta < t + np.timedelta64(nb_delta,
                                                             delta)]
        ind_delta = ind_delta[t_delta < t + np.timedelta64(nb_delta,
                                                           delta)]
        t_delta = t_delta[t_delta < t + np.timedelta64(nb_delta, delta)]

        bin_time, bin_sig,\
            bin_ind, _ = remove_disconnection_multi(t_delta, resp_delta,
                                                    ind_delta, fs)
        if bin_time[0].tolist():
            if is_list_of_list(bin_sig):

                times_delta.append(str(bin_time[0][0])[5: 16])
                peaks = detect_breathing_unwrap2(bin_sig, window_peaks=4,
                                                 amp_min=amp)
                rr, _ = remove_peaks_unwrap(peaks, bin_ind, fs)
            else:
                times_delta.append(str(bin_time[0])[5: 16])
                peaks = detect_breathing2(bin_sig, window_peaks=4,
                                          amp_min=amp)
                rr, _ = remove_peaks(peaks, bin_ind, fs)
                rr_delta.append(rr)
            if rr:
                bpm_delta.append(60 / np.mean(rr))
                hrv_delta.append(np.std(rr) * 1000)
            else:
                bpm_delta.append([])
                hrv_delta.append([])
        t = t + np.timedelta64(nb_delta, delta)
    return rr_delta, bpm_delta, hrv_delta, times_delta

def compute_deltas_ecg(timeline, sig, indicators_, nb_delta, delta,
                       fs, device_model_):
    t = timeline[0]
    #heart rate info
    rr_delta = []
    bpm_delta = []
    hrv_delta = []
    indicators_delta = []
    # breathing rate info using ecg_derived_respiration (edr)
    brr_delta = []
    cpm_delta = []
    times_delta = []
    while t < timeline[-1]:
        t_delta = timeline[timeline > t]
        ecg_delta = sig[timeline > t]
        ind_delta = indicators_[timeline > t]

        ecg_delta = ecg_delta[t_delta < t + np.timedelta64(nb_delta,
                                                           delta)]
        ind_delta = ind_delta[t_delta < t + np.timedelta64(nb_delta,
                                                           delta)]
        t_delta = t_delta[t_delta < t + np.timedelta64(nb_delta, delta)]

        bin_time, bin_sig,\
            bin_ind, _ = remove_disconnection_multi(t_delta, ecg_delta,
                                                    ind_delta, fs)
        unwrap_bin_ind = unwrap(bin_ind)
        if bin_time[0].tolist():
            indicators_delta.append(100*len(
                np.where(np.array(unwrap_bin_ind) == 1)[0]
                                      )/len(unwrap_bin_ind))
            if device_model_ == 'brassiere':
                if is_list_of_list(bin_sig):
                    times_delta.append(str(bin_time[0][0])[5:16])
                    peaks = detect_qrs_brassiere_unwrap(bin_sig, fs)
                    EDR_breathings = detect_breathing_EDR_unwrap(bin_sig,
                                                                 peaks,
                                                                 bin_ind, fs)
                    rr, _ = remove_peaks_unwrap(peaks, bin_ind, fs)
                else:
                    times_delta.append(str(bin_time[0])[5:16])
                    peaks = detect_qrs_brassiere(bin_sig, fs)
                    EDR_breathings = detect_breathing_EDR(bin_sig, peaks,
                                                          bin_ind, fs)
                    rr, _ = remove_peaks(peaks, bin_ind, fs)
            else:
                if is_list_of_list(bin_sig):
                    times_delta.append(str(bin_time[0][0])[5:16])
                    peaks = detect_qrs_unwrap(bin_sig, fs, mdp=0.55)
                    EDR_breathings = detect_breathing_EDR_unwrap(bin_sig,
                                                                 peaks,
                                                                 bin_ind, fs)
                    rr, _ = remove_peaks_unwrap(peaks, bin_ind, fs)

                else:
                    times_delta.append(str(bin_time[0])[5:16])
                    peaks = detect_qrs(bin_sig, fs, mdp=0.55)
                    EDR_breathings = detect_breathing_EDR(bin_sig, peaks,
                                                          bin_ind, fs)
                    rr, _ = remove_peaks(peaks, bin_ind, fs)
            rr_delta.append(rr)
            brr_delta.append(EDR_breathings)
            if rr:
                bpm_delta.append(60/np.mean(rr))
                hrv_delta.append(np.std(rr) * 1000)
            else:
                bpm_delta.append([])
                hrv_delta.append([])
            if EDR_breathings:
                cpm_delta.append(60/np.mean(EDR_breathings))
            else:
                cpm_delta.append([])
            t = t + np.timedelta64(nb_delta, delta)
    return rr_delta, bpm_delta, hrv_delta, times_delta, cpm_delta, brr_delta,\
               indicators_delta
