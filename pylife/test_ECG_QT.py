# -*- coding: utf-8 -*-


"""
Created on Thu May 19 10:57:30 2022

@author: Blandinelandrieu
"""

#%% libs
path_root = 'C:/Users/blandrieu/OneDrive - Passage innovation/Documents/GitHub/'
import sys
sys.path.append(path_root)

import os
os.chdir(path_root)
print(os.getcwd())

from pylife.env import get_env
DEV = get_env()
from pylife.datalife import Apilife
import matplotlib.pyplot as plt
import time
import numpy as np
#import numpy as np

#%% load data into al of apilife class
# Path file for API ids
path_ids =  'C:/Users/blandrieu/OneDrive - Passage innovation/Documents/GitCode/api' #'/home/claire/Desktop/prodv2'
path_save = path_ids+'saveTestsResults/'
name = "4veCS1_1801"

# Create dataframe containing relevant indicators
columnss = ['ts', 'Steps', 'RR', 'Min rr', 'Max rr', 'Var rr', 'pnn50', 'Bpm', 'Min bpm', 'Max bpm', 'Var bpm',
            'RR b1', 'Min rr b1', 'Max rr b1', 'Var rr b1','Rpm b1', 'Min rpm b1', 'Max rpm b1', 'Var rpm b1',
           'RR b2', 'Min rr b2', 'Max rr b2', 'Var rr b2', 'Rpm b2', 'Min rpm b2', 'Max rpm b2', 'Var rpm b2', 
           'Mean temp1', 'Mean temp2']
  
cecg = []
cbreath = []


# # Request info : end user avec stop de signal
# end_user        = '7fbCwc'
# from_time       = "2022-06-06 20:41:00"
# to_time         = "2022-06-06 20:45:00" # 22:45
# time_zone       = 'CEST'


# Request info : end user QT nul (signal très bruité)
end_user        = '3CqZP1'
from_time       =  "2023-02-08 20:00:00"
to_time         =  "2023-02-08 20:03:00" # 22:45
time_zone       = 'UTC'


activity_type = 'Test'
project = ''
df_all = []

params = {'path_ids': path_ids, 'end_user': end_user, 
         'from_time': from_time, 'to_time': to_time, 'time_zone': time_zone,
         'device_model': 't-shirt',
         'flag_acc': True, 'flag_breath': True,
         'flag_ecg': True, 'flag_temp': True, 'flag_temp_valid': True,
         'flag_imp': False, 
         'activity_types': activity_type, 'api_version':2,
         }
df_ = []

al = Apilife(params)
begin = time.time() 
al.get()
end = time.time()
print('Time taken for get:',round(end-begin,2))
begin = time.time() 
al.parse()
end = time.time()
print('Time taken for parse:',round(end-begin,2))
begin = time.time() 
al.filt()
end = time.time()
print('Time taken for filt:',round(end-begin,2))
begin = time.time() 
al.clean()
end = time.time()
print('Time taken for clean:',round(end-begin,2))
begin = time.time() 
al.analyze()
end = time.time()
print('Time taken for analyze:',round(end-begin,2))
# from_time = to_time
# to_time = str(np.datetime64(from_time)+np.timedelta64(5*60, 's')).replace("T", " ")

#%% values

print('qt_length_mean',  al.ecg.qt_length_mean_,'\n',
      'qt_length_',  al.ecg.qt_length_,'\n',
      'qt_length_std_',  al.ecg.qt_length_std_,'\n',
      'qt_length_median_',  al.ecg.qt_length_median_,'\n',
      'qt_length_median_clean_',  al.ecg.qt_length_median_corrected_,'\n'
      'qt_c_framingham' ,  al.ecg.qt_c_framingham_,'\n'
      'qt_c_framingham_per_seg' ,  al.ecg.qt_c_framingham_per_seg_
      )



#other results in order to check their shapes and types
test_q_start_ind =  al.ecg.q_start_index_
test_q_start_time =  al.ecg.q_start_time_
test_t_stop_ind =  al.ecg.t_stop_index_
test_t_stop_time =  al.ecg.t_stop_time_
test_t_qt_len =  al.ecg.qt_length_unwrap_
test_t_qt_len_clean =  al.ecg.qt_length_

       
print('first five QT lenghths in ms', test_t_qt_len[:5])

#%% qt_lenghts repartitions : check if QT length is stable and if there are improbable values 
# (above 600ms) in parts of the data
fig = plt.figure()
qt_lenght_unwrap         = al.ecg.qt_length_unwrap_



plt.plot(qt_lenght_unwrap, label= 'QT lenghts')
plt.plot([al.ecg.qt_length_mean_ for i in range(len(qt_lenght_unwrap))], label='mean', c='g')
plt.plot([al.ecg.qt_length_median_ for i in range(len(qt_lenght_unwrap))], label='median', c='black')
plt.plot([440 for i in range(len(qt_lenght_unwrap))], label='borderline QT prolongation 440ms', c='r', ls='--')
plt.plot([470 for i in range(len(qt_lenght_unwrap))], label='99th% for non ill males : QT prolongation of 470ms', c='cyan', ls='--')
plt.plot([480 for i in range(len(qt_lenght_unwrap))], label='99th% for non ill females : QT prolongation of 480ms', c='pink', ls='--')
plt.legend()
plt.title('QT lenght values in ms though time')


plt.show()

#%% QT lenghts histogram /!\ not corrected QT here


qt_lengths_list             = np.array(al.ecg.qt_length_unwrap_)
qt_lenght_unwrap_int_clean  = [qti for qti in qt_lengths_list if 250<qti<600]
qt_clean_median             = np.array(al.ecg.qt_length_median_corrected_)
                                       

fig = plt.figure()
plt.hist(qt_lenght_unwrap_int_clean, label= 'QT lenghts', bins=200)
plt.axvline(x= np.array(al.ecg.qt_length_median_) , color='black', linestyle='--', linewidth=3,  
            label='median QT length : ' +str(al.ecg.qt_length_median_[0])+'ms')
plt.axvline(x=qt_clean_median, color='cyan', linestyle='dotted', linewidth=3,  
            label='median on QT length between 350 and 550ms : ' +str(qt_clean_median)+'ms')
plt.axvline(x=440, color='orange', linestyle='dashed', linewidth=2,  
            label='borderline QT prolongation : 440ms')
plt.axvline(x=470, color='r', linestyle='dashed', linewidth=2,  
            label='99th% for non ill males : QT prolongation of 470ms')
plt.axvline(x=480, color='darkred', linestyle='dashed', linewidth=2,  
            label='99th% for non ill females : QT prolongation of 480ms')
plt.legend()
plt.title('QT lenght histogramme in ms')
plt.show()



#%% plot of start and stop on the clean signals (zoom for better visibility)

start_time              = al.ecg.q_start_time_
stop_time               = al.ecg.t_stop_time_
fig = plt.figure()
for i in range(len(start_time)):
    
    plt.plot(al.ecg.times_clean_2_[i] ,al.ecg.sig_clean_2_[i])
    plt.scatter(start_time[i], 
                [1 for i in range(len(start_time[i]))], c='b') 
    plt.scatter(stop_time[i], 
                [1 for i in range(len(stop_time[i]))], c='r')

#adding the legend just once
i                   = 0
plt.scatter(start_time[i], 
            [1 for i in range(len(start_time[i]))], c='b', label = 'start Q') 
plt.scatter(stop_time[i], 
            [1 for i in range(len(stop_time[i]))], c='r', label = 'stop T')
plt.legend()
plt.title('start and stop of QT on clean signal')

print('median', al.ecg.qt_length_median_)
print('mean', al.ecg.qt_length_mean_)
print('std', al.ecg.qt_length_std_)


