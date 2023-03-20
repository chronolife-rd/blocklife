# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:14:01 2023

@author: blandrieu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   scipy.ndimage import uniform_filter1d
import scipy.signal


#### support functions

def signal_smooth(signal, size=10):
    """**Signal smoothing**
    method="convolution", kernel="boxzen"
    Signal smoothing can be achieved using a filter kernel with the input
    signal to compute the smoothed signal (Smith, 1997)
    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    
    size : int
        Size of the kernel
   
    Returns
    -------
    array
        Smoothed signal.
    See Also
    ---------
    References
    ----------
    * Smith, S. W. (1997). The scientist and engineer's guide to digital signal processing.
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    length = len(signal)

    # Check length.
    size = int(size)
    if  size < 1:
        raise TypeError(
            "error: signal_smooth(): 'size' should be between 1 and length of the signal."
        )
    if size > length :
        smoothed = uniform_filter1d(signal, length, mode="mirror")
        return smoothed
    # Convolution
    # computes int(mean(window)) for a sliding window           
    # where the sides are filed with the miror of the signal 
    smoothed = uniform_filter1d(signal, size, mode="mirror") 

    return smoothed

def unwrap_sig (siglist : list):
    sigunwrap = list([])
    for si in siglist:
        #print("si", si[:10])
        #print("sigunwrap avt", sigunwrap[:10])
        sigunwrap = [*sigunwrap , *si]
        #print("sigunwrap apres", sigunwrap[:10])
    return sigunwrap


def keep_best_min_dist(peaks_list, mindist, valor):
    """when comparing two peaks with distance less than mindist,
    keep the peak with the greater valor value"""
    
    remaining_peaks = peaks_list.copy()
    remaining_valor = valor.copy()
    i = 0
    while( i <(len(remaining_peaks)-1)):
        #print('i' ,  i ,'peaks_list[i]', len(remaining_peaks),'valor[i]', remaining_valor[i] )
        if (remaining_peaks[i+1]-remaining_peaks[i])>mindist:
            i=1+i
        else:
            if remaining_valor[i]>remaining_valor[i+1]:
                remaining_peaks.remove(remaining_peaks[i+1])
                remaining_valor.remove(remaining_valor[i+1])
            else:
                remaining_peaks.remove(remaining_peaks[i])
                remaining_valor.remove(remaining_valor[i])
                
    return remaining_peaks



####  algos to add back missed peaks

def compare_sig_patern(signal, patern):
    """
    Parameters
    ----------
    signal : list or array
        signal (ECG for example)
    patern : list or array
        small segment of signal, a QRS for example.

    Returns
    -------
    dist : TYPE
        DESCRIPTION.
    exple : 
        # sigsin = [np.sin(x*3)*np.sin(x*0.5) for x in range(50)]
        # tempsin = sigsin[2:5]
        # comp = compare_sig_patern(signal = sigsin, 
        #                           patern = tempsin
        #                           )
        # plt.figure()
        # plt.plot(range(len(sigsin)),sigsin, c='b')
        # plt.plot(range(len(tempsin)),tempsin, c='r')
        # plt.plot(range(len(comp)),comp, c='g')
        # plt.show()

    """
    l = int(len(patern)/2)
    dist = []
    for i in range(len(signal)-len(patern)):
        i = i+l
        sigi = np.array(signal[i-l:i-l+len(patern)])
        #euclidian distance of pattern to signal of same length
        diff = np.sqrt(np.sum(np.square(sigi-np.array(patern))))
        dist.append(diff)
    dist = unwrap_sig([[dist[0] for i in range(l)], dist])  
    dist = unwrap_sig([dist, [dist[-1] for i in range(l)]])
    #print(dist)
    return dist


        
def find_missing_peaks (peaks_indexes, coef = 1.5):
    """
    Parameters
    ----------
    peaks_indexes : index list
        .
    coef : TYPE, float
        The default is 1.5.

    Returns
    -------
    dict
        return a dictionnary of the index start and index stop of RR wider than 
        coef*RR_median, also returns RR_median in the dictionnary.

    """
    peaks_indexes = np.array(peaks_indexes)
    rr = np.array( peaks_indexes[1:] - peaks_indexes[:-1])
    rr_med = np.median(rr)
    #rr_med = np.mean(rr)
    rr_ind_wide   = np.array([x for x in range(len(rr)) if rr[x]>rr_med*coef])
    #print(rr_ind_wide)
    miss_start = []
    miss_stop = []
    
    if len(rr_ind_wide)>0:
        miss_start    = np.array(peaks_indexes)[rr_ind_wide]
        miss_stop     = np.array(peaks_indexes)[rr_ind_wide+1]
        #print(miss_start, miss_stop)
    return {'start':miss_start, 'stop': miss_stop, 'rr_mean':rr_med}

def derivlist(l1):
    return np.array([l1[i]-l1[i+1] for i in range(len(l1)-1)])

def compare_missingpeak(sig, peaks_indexes, miss_start_ind, miss_stop_ind, QRSwindow = int(200*0.1)):
    patern1 = sig[int(miss_start_ind - QRSwindow) : int(miss_start_ind + QRSwindow)]
    patern2 = sig[int(miss_stop_ind  - QRSwindow) : int(miss_stop_ind  + QRSwindow)]
   
    compa1 = compare_sig_patern(derivlist(sig[miss_start_ind:miss_stop_ind]), 
                                patern = derivlist(patern1)
                                )
    compa2 = compare_sig_patern(derivlist(sig[miss_start_ind:miss_stop_ind]), 
                                patern = derivlist(patern2)
                                )

    if len(compa1)!=len(compa2):
        imax = min(len(compa1), len(compa2))
        return 0.5*(np.array(compa1[:imax-1])+np.array(compa2[:imax-1]))
    else :
        return 0.5*(np.array(compa1)+np.array(compa2))

def callback_missingpeaks (sig, peaks, fs = 200, show = False, coefmisspeak =1.5):
    qrs_len = int(200*0.1)
    peaks = np.array([qrs_len, *peaks, len(sig)-1-qrs_len])   ##############################
    newpeaks = list(peaks.copy())
   
    newlen = 0
    missingnb = 1
    while (newlen != len(newpeaks))&(missingnb>0):
        newlen = len(newpeaks)
        #print('newlen', newlen)
        missing = find_missing_peaks(peaks_indexes = newpeaks, coef = coefmisspeak)
        #print('bpm', round(200*60/(missing['rr_mean']),1), round(missing['rr_mean']*0.4,2))
        missingnb = len(missing['start'])
        
        if len(missing['start'])>0:
            
            for j in range(len(missing['start'])):
                comp = compare_missingpeak(sig = sig, 
                                           peaks_indexes = newpeaks,
                                           miss_start_ind = missing['start'][j], 
                                           miss_stop_ind = missing['stop'][j],
                                           QRSwindow = fs*0.05)
                
                coefrr = 0.5
                new_peak = _ecg_findpeaks_neurokit_fromdist(
                    signal = derivlist(comp),
                    sampling_rate=fs,
                    smoothwindow=0.1,
                    avgwindow=0.75,
                    gradthreshweight=1.4, #1.5
                    minlenweight=0.3, #4,
                    mindelay=missing['rr_mean']*0.4/fs, #0.2, #0.3
                    minlenght = 0.05,#0.05 ### additional conditions on the length of the QRS, twice smaller than smoothwindow    
                    show=False,
                    titlegraph = str(j),
                    minamplweight = 0.3,#0.2,
                    shortpeak = True)#True)
                
                for p in new_peak: # verify that the peaks found are not too close to the peaks around
                    if (p+missing['start'][j]+2 > missing['start'][j]+2 + coefrr*missing['rr_mean']) \
                        & (p+missing['start'][j]+2 < missing['stop'][j]+2 - coefrr*missing['rr_mean']):
                        newpeaks.append(p+missing['start'][j]+2)
                newpeaks.sort()
                newpeaks =list(np.unique(newpeaks))
                
        
    newpeaks.sort()
    newpeaks.remove(qrs_len)
    newpeaks.remove(len(sig)-1-qrs_len)
    
    return newpeaks
                        
#### supression of isolated extra peaks
def detect_second_doble (badpeak_df):
    
    badpeak_df['ind']    = badpeak_df.index
    badpeak_df['ind_p1'] = badpeak_df['ind'].shift(1)
    badpeak_df['ind_diff'] =  badpeak_df['ind'] -badpeak_df['ind_p1']
    badpeak_df['second_of_doble'] =0
    badpeak_df.loc[badpeak_df['ind_diff']==1, 'second_of_doble'] = 1
    
    return  {'second' : badpeak_df.loc[badpeak_df['second_of_doble'] ==1, 'peaks'].values, 
             'others' :badpeak_df.loc[badpeak_df['second_of_doble'] ==0, 'peaks'].values}

    

def supress_badpeaks(peaks_time, beta = 0.3):
    """
    Suppress peaks that are between two regular peaks:
        ( +     +     + +   +     +    +) -> ( +     +     +     +     +    +)
    Indeed it is unlikely that a cardiac anomaly can appear without disrupting 
    at all the beats before and after and the time between those two
    
    Limitations : 
    In the case where two beats verify this hypothesis and are very close, 
    we chose arbitrarilly to keep the second one
    
    This cleaning works when there is a badly annoted peak in the center of a sequence
    of six beats with near constant rr values. It doesn't work if there are mutiple 
    consecutive extra beats and for the three first and three last beats of a sequence
    
    For a difference between each rr of 50 ms (what is measured in pnn50), and a beat of 100bpm, 
    we have variations between the 2 first rr and the rr with the addditional peak in the middle of
    2*50ms/(60s/100bpm) -> 100ms/600ms -> 16%  to work in the case of some big heartbeat variations, 
    the beta should be around 2*16%
    
    
    """
    
    startlen = 0
    bad_peaks = []
    keep_beats = peaks_time.copy()
    while (startlen != len(keep_beats)):
        startlen = len(keep_beats)
        df=pd.DataFrame({'peaks' : keep_beats })
        df['peaks_m1'] = df['peaks'].shift(+1)
        df['peaks_p1'] = df['peaks'].shift(-1)
        df['rr']       = df['peaks'] - df['peaks_m1'] 
        df['2rr']      = df['peaks_p1'] - df['peaks_m1'] #sum of two consecutive rr
    
        df['2rr_p2']   = df['2rr'].shift(2)
        df['2rr_m2']   = df['2rr'].shift(-2)
    
        df['bad_peak'] = 0
        
        df.loc[(df['2rr_p2'] > 2*df['2rr']*(1-beta))
               &(df['2rr_p2'] < 2*df['2rr']*(1+beta))
               &(df['2rr_m2'] > 2*df['2rr']*(1-beta))
               &(df['2rr_m2'] < 2*df['2rr']*(1+beta))
               ,'bad_peak']=1
        df.loc[0,'bad_peak'] =0
        df.loc[df.shape[0]-1,'bad_peak'] =0
        
        keep_beats = df.loc[df['bad_peak'] ==0, 'peaks'].values
        
        second = detect_second_doble (df.loc[df['bad_peak']==1,['peaks','bad_peak']] )['second']
        
        # print('second', second)
        bad_peaks.append( detect_second_doble (df.loc[df['bad_peak']==1,['peaks','bad_peak']] )['others'])
        keep_beats = list(keep_beats) + list(second)
        keep_beats = sorted (keep_beats)
        keep_beats = list(np.unique(keep_beats))
        startlen   = len(keep_beats)
        
    return {'keep': keep_beats, 'bad':bad_peaks} 

#print(supress_badpeaks([0, 5, 10,12, 15, 20, 25, 27, 30, 35, 40 ], beta = 0.3))

#### find peaks on a small segment (60s or around this value)
  
def _ecg_findpeaks_neurokit_custom(
    signal,
    sampling_rate=200,
    smoothwindow=0.1,
    avgwindow=0.75,
    gradthreshweight=1.5,
    minlenweight=0.4,
    mindelay=0.2, #0.3
    minlenght = 0.05, ### additional conditions on the length of the QRS, twice smaller than smoothwindow    
    show=False,
    titlegraph = '',
    minamplweight = 0.2,
    shortpeak = True,
    prominence = 0,  #0.3
    ):
    """All tune-able parameters are specified as keyword arguments.
    The `signal` must be the highpass-filtered raw ECG with a lowcut of .5 Hz.
    """
    if show is True:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # Compute the ECG's gradient as well as the gradient threshold. Run with
    # show=True in order to get an idea of the threshold.
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = signal_smooth(absgrad,  size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad, size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))

    if show is True:
        
        ax1.plot(signal, label = 'signal')
        ax1.legend()
        ax2.plot(smoothgrad, label = 'smoothgrad')
        ax2.plot(avggrad, label = 'avggrad')
        ax2.plot(gradthreshold, label ='gradthreshold = '+str(gradthreshweight)+' * avggrad')
        ax2.legend()
        ax1.set_title(titlegraph + ' qrs = smoothgrad > gradthreshold')
        plt.legend()
        plt.show()

    # Identify start and end of QRS complexes.
    qrs = smoothgrad > gradthreshold
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    
    ### stop here if there are no peaks
    if len(beg_qrs)==0:
        return []
    
    # Throw out QRS-ends that precede first QRS-start.
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]
    #print('0 last end of QRS',end_qrs[-1] )

    # Identify R-peaks within QRS (ignore QRS that are too short).
    num_qrs = min(beg_qrs.size, end_qrs.size)
    #min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
    
    ######## modification to supress unlikely short QRSs
    widths = end_qrs[:num_qrs] - beg_qrs[:num_qrs]
    widths = [x for x in widths if x > minlenght*sampling_rate]
    min_len = np.median(widths) * minlenweight
    #print('min_len', min_len)
    #print('min_len mean ', np.mean(widths) * minlenweight)
    ######"#

    
    peaks = [0]
    shortQRS = []
    mindelayPeak = []
    ampl = gradthreshold - smoothgrad
    absampl = [abs(a) for a in ampl]
    
    for i in range(num_qrs):

        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        # if len_qrs < min_len:
        #     continue
    
        ######## modification t supress unlikely short QRSs
        if (len_qrs < min_len) or (len_qrs < minlenght*sampling_rate):
            if show is True:
                ax2.axvspan(beg, end, facecolor="b", alpha=0.5)
                shortQRS.append(beg)
            continue

        # Find local maxima and their prominence within QRS.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(prominence, None))
        
        if locmax.size < 1: #poor prominence
            if show is True:
                ax2.axvspan(beg, end, facecolor="orange", alpha=0.5)
            
        if locmax.size > 0: #ie if there is one point with a null derivative in this segment
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            peaks.append(peak)
            if show is True:
                ax2.axvspan(beg, end, facecolor="m", alpha=0.5)
            
    peaks.pop(0)
    #print('last peak after surpress signal sans prominnence', peaks[-1])
    
    ### stop here if there are no peaks
    if len(peaks)==0:
        return []
    
    ######### modification, supress QRS that lack amplitude of abs derivative
    if shortpeak == True:
        min_ampl = np.median(np.array(absampl)[peaks]) * minamplweight
        short_peaks= [p for p in peaks  if absampl[p]<min_ampl]
        peaks = [p for p in peaks  if absampl[p]>min_ampl]
        
        # do it a second time with an updated min_ampl that has supressed the first smallest peaks
        min_ampl = np.median(np.array(absampl)[peaks]) * minamplweight
        short_peaks= unwrap_sig([short_peaks, [p for p in peaks  if absampl[p]<min_ampl]])
        peaks = [p for p in peaks  if absampl[p]>min_ampl]
        
        #### to add, supress QRS that lack amplitude of QRS vals
        
    #######
    #print('last peak after surpress small ampl', peaks[-1])
    
    # Enforce minimum delay between peaks with the remaining peaks
    closepeaks = [peaks[i] for i in range(len(peaks)-1) if ((peaks[i+1] - peaks[i]) < mindelay)]
    #print('closepeaks', closepeaks)
    #print('num peaks avant', len(peaks))
    
    peaks = keep_best_min_dist(peaks_list = peaks,
                               mindist = mindelay,
                               valor =absampl )
    
    # show reason of supression in graphs
    if show is True:
        #print('len peaks', len(peaks))
        ax1.scatter(peaks, [signal[p] for p in peaks], c="r")
        ax2.set_title(str(len(peaks))+' peaks')
        
        for m in mindelayPeak :    
            ax1.axvline(m,  c="g")
        if shortpeak == True:
            for s in short_peaks :
                ax1.axvline(s, c="y")
        # for q in shortQRS :    
        #     ax1.axvline(q, c="b")
        for cl in closepeaks :    
            ax1.axvline(cl, c="r")

    peaks = np.asarray(peaks).astype(int)  # Convert to int
    return peaks
       
def _ecg_findpeaks_neurokit_fromdist(
    signal,
    sampling_rate=200,
    smoothwindow=0.075, #1,
    avgwindow=0.75,
    gradthreshweight=1.4, #1.
    minlenweight=0.3, #0.4
    mindelay=0.15, #0.3
    minlenght = 0.05, ### additional conditions on the length of the QRS, twice smaller than smoothwindow    
    show=False,
    titlegraph = '',
    minamplweight = 0.2,
    shortpeak = True):
    """All tune-able parameters are specified as keyword arguments.
    The `signal` must be the highpass-filtered raw ECG with a lowcut of .5 Hz.
    """
    if show is True:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # Compute the ECG's gradient as well as the gradient threshold. Run with
    # show=True in order to get an idea of the threshold.
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = min(int(np.rint(avgwindow * sampling_rate)), len(signal))
    smoothgrad = signal_smooth(absgrad,size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad,  size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))
    #print('mindelay', mindelay)
    if show is True:
        
        ax1.plot(signal, label = 'signal')
        ax1.legend()
        ax2.plot(smoothgrad, label = 'smoothgrad')
        ax2.plot(avggrad, label = 'avggrad')
        ax2.plot(gradthreshold, label ='gradthreshold = '+str(gradthreshweight)+' * avggrad')
        ax2.legend()
        ax1.set_title(titlegraph + ' qrs = smoothgrad > gradthreshold')
        plt.legend()
        plt.show()

    # Identify start and end of QRS complexes.
    qrs = smoothgrad > gradthreshold
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    ### stop here if there are no peaks
    if len(beg_qrs)==0:
        return []
    
    # Throw out QRS-ends that precede first QRS-start.
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]
    #print('0 last end of QRS',end_qrs[-1] )

    # Identify R-peaks within QRS (ignore QRS that are too short).
    num_qrs = min(beg_qrs.size, end_qrs.size)
    #min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
    
    ######## modification to supress unlikely short QRSs
    widths = end_qrs[:num_qrs] - beg_qrs[:num_qrs]
    widths = [x for x in widths if x > minlenght*sampling_rate]
    ### stop here if there are no peaks
    if len(widths)==0:
        return []
    min_len = max(widths) * minlenweight
    #print('min_len', min_len)
    #print('min_len mean ', np.mean(widths) * minlenweight)
    ######"#

    
    peaks = [0]
    shortQRS = []
    mindelayPeak = []
    ampl = gradthreshold - smoothgrad
    absampl = [abs(a) for a in ampl]
    
    for i in range(num_qrs):

        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        ### modification t supress unlikely short QRSs
        if (len_qrs < min_len) or (len_qrs < minlenght*sampling_rate):
            if show is True:
                ax2.axvspan(beg, end, facecolor="b", alpha=0.5)
                shortQRS.append(beg)
            continue

        # Find local maxima and their prominence within QRS.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(0.3, None))

        if locmax.size > 0: #ie if there is one point with a null derivative in this segment
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            peaks.append(peak)
            if show is True:
                ax2.axvspan(beg, end, facecolor="m", alpha=0.5)
            
    peaks.pop(0)
    
    ### stop here if there are no peaks
    if len(peaks)==0:
        return []
    
    ### supress QRS that lack amplitude
    if shortpeak == True:
        min_ampl = max(np.array(absampl)[peaks]) * minamplweight
        short_peaks= [p for p in peaks  if absampl[p]<min_ampl]
        peaks = [p for p in peaks  if absampl[p]>min_ampl]
        
        # do it a second time with an updated min_ampl that has supressed the first smallest peaks
        min_ampl = np.median(np.array(absampl)[peaks]) * minamplweight
        short_peaks= unwrap_sig([short_peaks, [p for p in peaks  if absampl[p]<min_ampl]])
        peaks = [p for p in peaks  if absampl[p]>min_ampl]
    
    
    ### Enforce minimum delay between peaks with the remaining peaks
    ##### par rapport au permier et dernier peak
    #print('peaks, mindelay, lensig', peaks, mindelay , len(signal))
    peaks = [peaks[i] for i in range(len(peaks)) if (peaks[i] > mindelay)&(peaks[i] < (len(signal)- mindelay))]
    #print('peaks keep', peaks )
    
    ##### entre les nouveaux pics détectés
    if len(peaks)>0:
        closepeaks = [peaks[i] for i in range(len(peaks)-1) if ((peaks[i+1] - peaks[i]) < mindelay)]
                   
        peaks = keep_best_min_dist(peaks_list = peaks,
                                   mindist = mindelay,
                                   valor =absampl )

    ### show reason of supression in graphs
    if show is True:
        #print('len peaks', len(peaks))
        ax1.scatter(peaks, [signal[p] for p in peaks], c="r")
        ax2.set_title(str(len(peaks))+' peaks')
        
        for m in mindelayPeak :    
            ax1.axvline(m,  c="g")
        if shortpeak == True:
            for s in short_peaks :
                ax1.axvline(s, c="y")
        # for q in shortQRS :    
        #     ax1.axvline(q, c="b")
        for cl in closepeaks :    
            ax1.axvline(cl, c="r")

    peaks = np.asarray(peaks).astype(int)  # Convert to int
    return peaks



#### find peak indexes on ECG signal
def get_Peaks( sig, plot = False, fs = 200, mindistpeak=0.2, 
                      supress_extrap = False, missed_peak = True, prominence = 0, addindex = 0, minamplweight=0.2):
    """
    detection of peaks
    if the segment is smaller than 0.1s return []
    if the segment is longer than 80s, apply on 63s subsegments with
    a 3s overlap and supress redundant peak indexes
    """
    #print('whole ',  len(sig))
    
    if len(sig)<(fs*0.1):
        return []
   
    if len(sig)<(fs*71): # for the smoothing fuction to work well
        pred_peak  =_ecg_findpeaks_neurokit_custom(signal = sig, 
                                  sampling_rate= fs, 
                                  show = plot,
                                  mindelay=mindistpeak,
                                  prominence = prominence,
                                  minamplweight = minamplweight)
    else :
        #recursive function to apply on 63s subsegments
        pred_peak = [*get_Peaks( sig = sig[:60*fs], plot = plot, fs = fs, mindistpeak=mindistpeak, 
                              supress_extrap = supress_extrap, missed_peak = missed_peak, prominence = prominence), \
                      *get_Peaks( sig = sig[57*fs:], plot = plot, fs = fs, mindistpeak=mindistpeak, 
                              supress_extrap = supress_extrap, missed_peak = missed_peak, prominence = prominence, addindex =57*fs)]
        #supress reduncdency that may have been acquired in overlapping segments   
        #print(len(pred_peak))
        pred_peak.sort()
        pred_peak =list(np.unique(pred_peak))
        #print(len(pred_peak))
        

    if supress_extrap == True:
        for i in range(4):
            pred_peak = supress_badpeaks(pred_peak, beta = 0.3)['keep']        
            
    if missed_peak == True :
        newlen = 0
        while newlen != len(pred_peak):
            newlen = len(pred_peak)
            pred_peak = callback_missingpeaks (sig = sig,
                                                peaks = pred_peak,
                                                fs = fs,
                                                show = plot,
                                                coefmisspeak = 1.4) ###coef a vérifer ntre 1.3, 1.4 et 1.5
    if plot==True:
        plt.figure()
        plt.plot(range(len(sig)),sig, '-', label = 'Signal')
        
        for k in range(len(pred_peak)):
            plt.axvline(pred_peak[k], c='darkblue', linestyle = '--')
        plt.axvline(pred_peak[0], c='darkblue', label='peak',  linestyle = '--')
        
        plt.legend()
        plt.title('Peaks')
        plt.plot()
        
    return  list(np.array(pred_peak)+[addindex])

def getPeaks_unwrap( sigs,fs = 200, mindistpeak=0.2, 
                      supress_extrap = False, missed_peak = True, 
                      prominence = 0, minamplweight=0.2,  plot = False):
    """
    sigs : list of ECG signals
    
    returns : list of peak indexes (indexed in each)
    
    /!\ misspeak = True works better on cleaned signal, else it migh be too 
    influenced by noisy parts and add bad peaks ressembling already incorrectly
    detected peaks
    
    /!\ supress extra peaks might supress extrasystoles, only use for SDNN or other 
    time when needing only good peaks
    """
    peaks_index = []
    addindex = 0
    for i, sig in enumerate(sigs):
        peaks_index.append(get_Peaks( sig, plot = plot , fs = fs , mindistpeak = mindistpeak, 
                              supress_extrap = supress_extrap, missed_peak = missed_peak, 
                              prominence = prominence, addindex = addindex, minamplweight=minamplweight))
    return peaks_index
        

def peak_ampl (sig, peak_indx, fs = 200):
    """
    QRS complex last less than 0.12s in non pathologic cases
    this function returns the min and max values in a 0.16s window centered on the peak
    in order to account for most QRS including pathologic ones
    
    
    exple
    x = range(1,8000)
    sinx = [np.sin(xx/80) for xx in x]
    indxes =[i*200 for i in range(int(8000/200))]

    sinampl = peak_ampl (sinx, indxes, fs = 200)

    plt.figure()
    plt.plot(x, sinx)
    plt.scatter(indxes,[0 for z in indxes], c='b' , s=8)
    plt.scatter(indxes,sinampl, c='r' , s=8, label = 'amplitude in interval')
    for p in indxes[1:]:
        plt.axvline(p-int(0.08*200))
    for p in indxes[:-1]:
        plt.axvline(p+int(0.08*200))
    plt.show()


    """
    ampls= []
    l = len(sig)
    
    # if there is no peak
    if len(peak_indx)<1:
        return ampls
    delta = int(0.08*fs)
    
    # if the signal is less or equal to 0.16s wide
    if l<(delta*2):
        return [max(sig) - min(sig)]

    #start
    sig_window =  sig[max(0,  peak_indx[0]) : peak_indx[0] + delta]
    ampls.append(max(sig_window) -  min(sig_window))
    
    #middle
    for p in peak_indx[1:-1]:
        sig_window =  sig[p - delta : p + delta] 
        ampls.append(max(sig_window) -  min(sig_window))
    
    #end
    sig_window =  sig[peak_indx[-1] : min(l, peak_indx[-1] + delta)]
    ampls.append(max(sig_window) -  min(sig_window))
        
    return ampls


def unwrap_peak_ampl (sigs, peak_indxs, fs = 200):
    return [peak_ampl (sigs[i], peak_indxs[i], fs = 200) for i in range(len(sigs))]

