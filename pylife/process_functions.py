from pylife.env import get_env
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    pass
# import numpy as np
import datetime
from pylife.siglife import Acceleration_x
from pylife.siglife import Acceleration_y
from pylife.siglife import Acceleration_z
from pylife.siglife import Accelerations
from pylife.siglife import Breath_1
from pylife.siglife import Breath_2
from pylife.siglife import Breaths
from pylife.siglife import ECG
from pylife.siglife import Temperature_1
from pylife.siglife import Temperature_2
from pylife.siglife import Temperatures
from pylife.datalife import Datalife
# from pylife.useful import compute_signal_mean
import math

def process_data_interval(dict_params):
    """ Process data for a given end_user in a time interval 
   input dict_params = {
       "ecg":       dict of sig info,
       "breath_1":  dict of sig info,
       "breath_2":  dict of sig info,
       "temp_1":    dict of sig info,
       "temp_2":    dict of sig info,
       "accx":      dict of sig info,
       "accy":      dict of sig info,
       "accz":      dict of sig info
       }   
       and dict of sig info = {times, sig, fs, fw_version}
   
    returns dict_results
   
    """
    
    keys = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
            'temp_1', 'temp_2']
    for key in keys:
        if key not in dict_params.keys():
            dict_params[key]  = {
                                    'times':        [], 
                                    'sig':          [], 
                                    'fs':           [],
                                    'fw_version':   []
                                }
        
    # Define parameters for init
    dict_params['init'] = {'flag_acc': True, 'flag_breath': True, 
                           'flag_ecg': True, 'flag_temp': True}
    
    # Init class for processing
    dl = Datalife(dict_params['init'])
    dl.init()
    dl.check(dict_params['init'])
    dl.assign(dict_params['init'])
    
    # Define signal classes
    dl.accx     = Acceleration_x(dict_params['accx'])
    dl.accy     = Acceleration_y(dict_params['accy'])
    dl.accz     = Acceleration_z(dict_params['accz'])
    dl.accs     = Accelerations(dl.accx, dl.accy, dl.accz)
    dl.breath_1 = Breath_1(dict_params['breath_1'])
    
    dl.breath_2 = Breath_2(dict_params['breath_2'])
    dl.breaths  = Breaths(dl.breath_1, dl.breath_2)
    dl.ecg      = ECG(dict_params['ecg'])
    dl.temp_1   = Temperature_1(dict_params['temp_1'])
    dl.temp_2   = Temperature_2(dict_params['temp_2'])
    dl.temps    = Temperatures(dl.temp_1, dl.temp_2)
    
    # Process
    dl.filt()
    dl.clean()
    dl.analyze()
    
    # Store results
    dict_result = {}
    
    if not dl.accs.is_empty_:
        if isinstance(dl.accs.mean_activity_level_, float):
            averaged_activity = round(dl.accs.mean_activity_)
        else:
            averaged_activity = 0
            
        dict_result["steps_number"]             = dl.accs.n_steps_
        dict_result["activity_level"]           = [item.tolist() for item in dl.accs.activity_level_]
        dict_result["averaged_activity"]        = averaged_activity
        dict_result["activity_level_times"]     =[item.tolist() for item in dl.accx.times_] #[time_sequence[0].astype(datetime.datetime).replace(tzinfo=datetime.timezone.utc)
 #                                                  for time_sequence in dl.accx.times_]
        
    
    if not dl.breath_1.is_empty_:
        rpm = None
        if len(dl.breath_1.rpm_) > 0:
            rpm      = dl.breath_1.rpm_[0]
            rpm      = int(round(rpm))
            rpm_s    = dl.breath_1.rpm_s_[0]
            rpm_s      = int(round(rpm_s))
           
        times, br_filtered, indic   = dl.breath_1.select_on_sig('filt')
        dict_result["respiratory_rate"]                 = rpm
        dict_result["breath_1_filtered"]                = [item.tolist() for item in dl.breath_1.sig_filt_]
        dict_result["respiratory_rate_quality_index"]   = 0 if rpm is None else 1
        dict_result["breath_quality_index"]             = dl.breath_1.indicators_seconds_
        dict_result["breath_1_filtered_times"]          = [item.tolist() for item in times] 
        dict_result["breath_quality_index_frequency"]   = dl.breath_1.indicators_frequency_
        dict_result["breath_1_rate"]                    = rpm_s # duree moyenne de la respiration (en secondes)
        
        #dict_result["breath_1_peaks_index"]             = dl.breath_1.peaks_
        #dict_result["breath_1_valleys_index"]           = dl.breath_1.valleys_ 
        dict_result["breath_1_peaks_times"]             = dl.breath_1.peaks_times_ 
        dict_result["breath_1_valleys_times"]           = dl.breath_1.valleys_times_ 
        
    
    if not dl.breath_2.is_empty_:
        rpm = None
        if len(dl.breath_2.rpm_) > 0:
            rpm      = dl.breath_2.rpm_[0]
            rpm      = int(round(rpm))
            rpm_s    = dl.breath_2.rpm_s_[0]
            rpm_s      = int(round(rpm_s))
            
        times, br_filtered, indic   = dl.breath_2.select_on_sig('filt')
        dict_result["respiratory_rate_abdominal"]       = rpm
        dict_result["breath_2_filtered"]                = [item.tolist() for item in br_filtered]
        dict_result["breath_2_filtered_times"]          = [item.tolist() for item in times]  #[time_sequence[0].astype(datetime.datetime).replace(tzinfo=datetime.timezone.utc)
#                                                           for time_sequence in times]
        dict_result["respiratory_rate_2_quality_index"] = 0 if rpm is None else 1
        dict_result["breath_2_rate"]                    = rpm_s # duree moyenne de la respiration (en secondes)
        
        #dict_result["breath_2_peaks_index"]             = dl.breath_2.peaks_
        #dict_result["breath_2_valleys_index"]           = dl.breath_2.valleys_ 
        dict_result["breath_2_peaks_times"]             = dl.breath_2.peaks_times_ 
        dict_result["breath_2_valleys_times"]           = dl.breath_2.valleys_times_
        
        

    if not dl.ecg.is_empty_:
        hr = None
        if len(dl.ecg.bpm_) > 0:
            hr      = dl.ecg.bpm_[0]
            hr      = int(round(hr))
            
        hrv = None
        if len(dl.ecg.bpm_var_ms_) > 0:
            hrv      = dl.ecg.bpm_var_ms_[0]
            hrv      = int(round(hrv))
            
        rr = None
        if len(dl.ecg.bpm_ms_) > 0:
            rr      = dl.ecg.bpm_ms_[0]
            rr      = int(round(rr))
            
        sdnn = None
        if len(dl.ecg.sdnn_) > 0:
            sdnn    = dl.ecg.sdnn_[0]
            sdnn    = int(round(sdnn))
            
        rmssd = None
        if len(dl.ecg.rmssd_) > 0:
            rmssd   = dl.ecg.rmssd_[0]
            rmssd   = int(round(rmssd))
            
        lnrmssd = None
        if len(dl.ecg.lnrmssd_) > 0:
            lnrmssd = dl.ecg.lnrmssd_[0]
            lnrmssd = int(round(lnrmssd))
            
        pnn50 = None
        if len(dl.ecg.pnn50_) > 0:
            pnn50   = dl.ecg.pnn50_[0]
            pnn50   = int(round(pnn50))
            
        r_peak_index = None
        if len(dl.ecg.peaks_) > 0:
            r_peak_time  = dl.ecg.peaks_times_
        
        q_start_time = None
        if len(dl.ecg.q_start_time_) > 0:
            q_start_time   = dl.ecg.q_start_time_
            
        t_stop_index = None
        if len(dl.ecg.t_stop_time_) > 0:
            t_stop_time   = dl.ecg.t_stop_time_
            
        qt_length_median_corrected = None
        if len(dl.ecg.qt_length_median_corrected_) > 0:
            qt_length_median_corrected   = dl.ecg.qt_length_median_corrected_[0]
            qt_length_median_corrected   = int(round(qt_length_median_corrected))
            
        dict_result["heartbeat"]                    = hr
        dict_result["HRV"]                          = hrv
        dict_result["ecg_filtered"]                 = [item.tolist() for item in dl.ecg.sig_filt_mv_]
        dict_result["heartbeat_quality_index"]      = 0 if hrv is None else 1
        dict_result["ecg_quality_index"]            = dl.ecg.indicators_seconds_
        dict_result["HRV_quality_index"]            = 0 if hrv is None else 1
        dict_result["ecg_filtered_times"]           = [item.tolist() for item in dl.ecg.times_]  
                                                 
        dict_result["ecg_quality_index_frequency"]  = dl.ecg.indicators_frequency_
        dict_result["rr_interval"]                  = rr
        dict_result["sdnn"]                         = sdnn
        dict_result["rmssd"]                        = rmssd
        dict_result["lnrmssd"]                      = lnrmssd
        dict_result["pnn50"]                        = pnn50
        
        dict_result["q_start_time"]                 = q_start_time  # lists of indexes for each clean segment, the index number restart from 0 at each segment 
        dict_result["r_peak_time"]                  = r_peak_time 
        dict_result["t_stop_time"]                  = t_stop_time
        dict_result["qt_length_median_corrected"]   = qt_length_median_corrected
        
        
    if not dl.temp_1.is_empty_:
        dict_result["averaged_temp_1"] = round(dl.temp_1.mean_*1e4)/1e2
        
    if not dl.temp_2.is_empty_:
        dict_result["averaged_temp_2"] = round(dl.temp_2.mean_*1e4)/1e2
        
    dict_result["is_worn"] = 1 if dl.is_worn_ else 0
    
    return dict_result