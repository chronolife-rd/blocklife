import numpy as np
from pylife.env import get_env
import random
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import pandas as pd
    import pickle
# --- Add imports for PROD and DEV env

def simul_acc(duration, fs, n_step_pm, noise_amp=0):

    fc = n_step_pm/60
    times = np.arange(0, duration, 1/fs)
    n = len(times)
    sig = np.sin(2*np.pi*times*fc)
    sig = sig + noise_amp*np.random.randn(n)

    return np.array(sig)

def simul_breath(duration, fs, rpm, noise_amp=0):

    fc = rpm/60
    times = np.arange(0, duration, 1/fs)
    n = len(times)
    sig = np.sin(2*np.pi*times*fc*2)
    sig += 1.5*np.sin(2*np.pi*times*fc)    
    sig = sig + noise_amp*np.random.randn(n)

    return np.array(sig)
    
def simul_ecg(duration, fs, bpm, afib=False, noise_amp=0):

    # Simul QRS complex with sinusoide
    fc = 10
    T = 1/fc
    times_qrs = np.arange(0, T, 1/fs)
    qrs = np.sin(2*np.pi*times_qrs*fc)

    # Simul BPM and RR intervals
    hr = 60/bpm
    n_rr = round(duration/hr)
    rr_s = np.repeat(hr, n_rr)
    if afib:
        noise = 1/5*np.random.randn(len(rr_s))
        rr_s = rr_s + noise
        while sum(rr_s) < duration:
            noise = 1/5*np.random.randn(1)
            rr_s = np.append(rr_s, hr + noise)
    # Init outputs
    times = np.arange(0, duration, 1/fs)
    n = len(times)
    sig = np.zeros(n)

    # Make signal
    imin = 0
    for rr in rr_s:
        rr_interval = round(rr*fs)
        if len(qrs) > rr_interval:
            rr_interval = len(qrs)*2

        imin += int(rr_interval)
        imax = imin + len(qrs)
        if imax > n:
            break
        sig[imin:imax] = qrs

    # Add noise
    sig = sig + noise_amp*np.random.randn(n)

    return np.array(sig)

def simul_temp(duration, fs, mean_temp, noise_amp=0):
    
    times = np.arange(0, duration, 1/fs)
    n = len(times)
    sig = mean_temp + noise_amp*np.random.randn(n)
    
    return np.array(sig)
    
def simul_imp(duration, fs):
    
    times = np.arange(0, duration, 1/fs)
    n = len(times)
    imp_min = 5
    imp_max = 80
    sig = random.sample(range(imp_min, imp_max), n)
    
    return np.array(sig)

def simul_noise(duration, fs, noise_amp):
    
    times = np.arange(0, duration, 1/fs)
    n = len(times)
    noise = noise_amp*np.random.randn(n)
    
    return noise
    
def get_sig_info(datas, signal_type):
    
    output = {}
    data = datas[signal_type]
    
    output['times'] = data.times_
    output['sig'] = data.sig_
    output['fs'] = data.fs_
    output['fw_version'] = '0705_0355'

    return output