import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
path_pylife = 'C:/Users/MichelClet/Desktop/Files/pylife'
sys.path.append(path_pylife)
path_data = 'C:/Users/MichelClet/Desktop/Files/data/simul/'
from pylife.data_manager.load_data import simul_breath, simul_ecg, simul_acc, simul_temp, simul_imp, simul_noise
from pylife.signal_processing.siglife import ECG, Breath_1, Temperature_1, Impedance_1
from pylife.signal_processing.siglife import Acceleration_x, Accelerations
# from pylife.signal_processing.useful import unwrap, wrap

# %%
flag_1seg_ = [True, False]
flag_disconnection_ = [True, False]
flag_clean_acc_ = [True, False]
flag_clean_breath_ = [True, False]
flag_clean_ecg_ = [True, False]
flag_clean_temp_ = [True, False]
disconnection_duration = np.timedelta64(10, 's') # seconds
duration = 1*60
sig_noise_amp = 8
time_now = np.datetime64('now')

n_step_pm = 100
rpm = 18
bpm = 60
for flag_1seg in flag_1seg_:
    if flag_1seg:
        n_seg = 1
    else:
        n_seg = 2
    for flag_disconnection in flag_disconnection_:
        for flag_clean_acc in flag_clean_acc_:
            for flag_clean_breath in flag_clean_breath_:
                for flag_clean_ecg in flag_clean_ecg_:
                    for flag_clean_temp in flag_clean_temp_:
                        # --------------------------------------------------------
                        # % ACC
                        fs=50
                        
                        sig_s = []
                        times_s = []
                        last_time = time_now
                        for i in range(n_seg):
                            if flag_clean_acc:
                                sig = simul_acc(duration, fs, n_step_pm, noise_amp=.5)
                                sig = sig*200
                                # sig2 = simul_acc(duration/2, fs, n_step_pm=0, noise_amp=.5)
                                # sig = np.hstack((sig1, sig2))
                            else:
                                sig = simul_noise(duration, fs, noise_amp=sig_noise_amp)
                            sig = sig.astype('int')
                            times = np.arange(last_time, 
                                              last_time + np.timedelta64(int(len(sig)/fs*1e6), 'us'),
                                              np.timedelta64(int(1/fs*1e6), 'us'))
                            last_time = times[-1]
                            if flag_disconnection:
                                last_time = last_time + disconnection_duration
                            sig_s.append(sig)
                            times_s.append(times)
                        accx = Acceleration_x({'times': times_s, 'sig': np.array(sig_s), 'fs': fs,
                                               'fw_version': '0'})
                        accy = Acceleration_x({'times': times_s, 'sig': np.array(sig_s)-20, 'fs': fs,
                                               'fw_version': '0'})
                        accz = Acceleration_x({'times': times_s, 'sig': np.array(sig_s)+20, 'fs': fs,
                                               'fw_version': '0'})
                        acc = Accelerations(accx, accy, accz)
                        acc.clean()
                        acc.analyze()
                        # plt.figure()
                        # accx.show()
                        print('N steps', acc.n_steps_)
                        
                        # --------------------------------------------------------
                        # % BREATH
                        fs=20
                        sig_s = []
                        times_s = []
                        last_time = time_now
                        for i in range(n_seg):
                            if flag_clean_breath:
                                sig = simul_breath(duration, fs, rpm, noise_amp=1e-1)
                                sig = sig*100
                            else:
                                sig = simul_noise(duration, fs, noise_amp=sig_noise_amp)
                            sig = sig.astype('int')
                            times = np.arange(last_time, 
                                              last_time + np.timedelta64(int(len(sig)/fs*1e6), 'us'),
                                              np.timedelta64(int(1/fs*1e6), 'us'))
                            last_time = times[-1]
                            if flag_disconnection:
                                last_time = last_time + disconnection_duration
                            sig_s.append(sig)
                            times_s.append(times)
                        breath = Breath_1({'times': times_s, 'sig': sig_s, 'fs': fs,
                                               'fw_version': '0'})
                        breath.clean()
                        breath.analyze()
                        print('RPM', np.mean(breath.rpm_))
                        # plt.figure()
                        # breath.show()
                        
                        # --------------------------------------------------------
                        # % ECG
                        noise = 1/50
                        fs = 200
                        prev = 1/100
                        sig_s = []
                        times_s = []
                        last_time = time_now
                        for i in range(n_seg):
                            if flag_clean_ecg:
                                sig = simul_ecg(duration, fs, bpm, afib=False, noise_amp=1e-3)
                                sig = sig*200
                            else:
                                sig = simul_noise(duration, fs, noise_amp=sig_noise_amp)
                            sig = sig.astype('int')
                            times = np.arange(last_time, 
                                              last_time + np.timedelta64(int(len(sig)/fs*1e6), 'us'),
                                              np.timedelta64(int(1/fs*1e6), 'us'))
                            last_time = times[-1]
                            if flag_disconnection:
                                last_time = last_time + disconnection_duration
                                
                            sig_s.append(sig)
                            times_s.append(times)
                        ecg = ECG({'times': times_s, 'sig': sig_s, 'fs': fs,
                                               'fw_version': '0'})
                        ecg.clean()
                        ecg.analyze()
                        print('BPM', np.mean(ecg.bpm_))
                        # plt.figure()
                        # ecg.show()
                        
                        # --------------------------------------------------------
                        # % TEMP
                        fs = 1
                        sig_s = []
                        times_s = []
                        last_time = time_now
                        for i in range(n_seg):
                            sig = simul_temp(duration, fs, mean_temp=25, noise_amp=.1)
                            times = np.arange(last_time, 
                                              last_time + np.timedelta64(int(len(sig)/fs*1e6), 'us'),
                                              np.timedelta64(int(1/fs*1e6), 'us'))
                            last_time = times[-1]
                            if flag_disconnection:
                                last_time = last_time + disconnection_duration
                            sig_s.append(sig)
                            times_s.append(times)
                        temp = Temperature_1({'times': times_s, 'sig': sig_s, 'fs': fs,
                                               'fw_version': '0'})
                        temp.clean()
                        temp.analyze()
                        # print('Temp mean', np.mean(temp.mean_))
                        # plt.figure()
                        # temp.show()
                        # plt.ylim(20,30)
                                
                        # --------------------------------------------------------
                        # % IMP
                        fs = 1/(10*60)
                        sig_s = []
                        times_s = []
                        last_time = time_now
                        for i in range(n_seg):
                            sig = simul_imp(duration, fs)
                            sig = sig.astype('int')
                            times = np.arange(last_time, 
                                              last_time + np.timedelta64(int(len(sig)/fs*1e6), 'us'),
                                              np.timedelta64(int(1/fs*1e6), 'us'))
                            last_time = times[-1]
                            if flag_disconnection:
                                last_time = last_time + disconnection_duration
                            sig_s.append(sig)
                            times_s.append(times)
                        imp = Impedance_1({'times': times_s, 'sig': sig_s, 'fs': fs,
                                               'fw_version': '0'})
                        # plt.figure()
                        # imp.show()
                        
                        # %
                        savefilename = ['1seg_','disco_', 'acc_', 'breath_', 'ecg_', 'temp']
                        if not flag_1seg:
                            savefilename[0] = 'no-' + savefilename[0]
                        if not flag_disconnection:
                            savefilename[1] = 'no-' + savefilename[1]
                        if not flag_clean_acc:
                            savefilename[2] = 'no-' + savefilename[2]
                        if not flag_clean_breath:
                            savefilename[3] = 'no-' + savefilename[3]
                        if not flag_clean_ecg:
                            savefilename[4] = 'no-' + savefilename[4]
                        if not flag_clean_temp:
                            savefilename[5] = 'no-' + savefilename[5]
                        savefilename = ''.join(savefilename)
                        data = {'accx': accx, 'accy': accy, 'accz': accz, 'acc': acc, 'breath': breath, 
                                'ecg': ecg, 'temp': temp, 'imp': imp}
                        with open((path_data + savefilename + '.pkl'), 'wb') as file:
                            pickle.dump(data, file)