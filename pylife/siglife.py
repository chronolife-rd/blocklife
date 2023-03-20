import numpy as np
from random import randint
from pylife.useful import unwrap
from pylife.useful import is_list_of_list
from pylife.useful import get_durations_info #, get_sig_loss
from pylife.useful import get_stats_clean_sig
from pylife.useful import get_stats_not_clean_sig
from pylife.useful import get_peaks_from_peaks_times
#from pylife.useful import get_peaks_amps_unwrap
from pylife.useful import get_peaks_times_unwrap
from pylife.useful import set_list_of_list
from pylife.useful import count_usable_segments
from pylife.useful import convert_mv_unwrap
from pylife.useful import transform_indicators_seconds
from pylife.useful import compute_rr_features_unwrap
from pylife.useful import compute_rsp_features_unwrap
from pylife.useful import compute_qt_times
from pylife.useful import fw_version_test
from pylife.useful import get_rsp_peaks_clean_unwrap, get_peaks_clean_unwrap
from pylife.useful import compute_median_iqr_window_unwrap
#from pylife.compute_deltas import compute_deltas_breath
#from pylife.compute_deltas import compute_deltas_ecg
from pylife.remove import remove_noise_peak_valley_unwrap, remove_saturation_and_big_ampls_unwrap
from pylife.remove import remove_no_rsp_signal_unwrap
from pylife.remove import remove_noise_unwrap, remove_noise
from pylife.remove import remove_noise_with_emd_unwrap
#from pylife.remove import remove_noise_with_psd_unwrap
from pylife.remove import remove_noise_smooth_unwrap
from pylife.remove import remove_noisy_temp_unwrap
# from pylife.remove import remove_noisy_temp
# from pylife.remove import remove_peaks_unwrap
# from pylife.remove import remove_peaks
# from pylife.remove import remove_timestamps
from pylife.remove import remove_timestamps_unwrap
# from pylife.remove import remove_disconnection_multi
# from pylife.remove import remove_false_rr
# from pylife.filters import filter_ecg_unwrap
from pylife.filters import filter_ecg_scipy_unwrap
from pylife.filters import filter_breath_unwrap
#from pylife.filters import filter_breath
from pylife.filters import filter_acceleration_unwrap
from pylife.filters import filter_acceleration
# from pylife.cyclic_measurement import compute_ppm_unwrap
# from pylife.cyclic_measurement import compute_ppm
from pylife.cyclic_measurement import pos_2_time_interval_unwrap
# from pylife.cyclic_measurement import pos_2_time_interval
# from pylife.detection import detect_breathing
# from pylife.detection import detect_breathing_unwrap
# from pylife.detection import detect_breathing2
# from pylife.detection import detect_breathing_unwrap2
from pylife.detection import detect_qrs_unwrap
#from pylife.detection import detect_qrs
from pylife.detection import detect_peaks_unwrap
from pylife.detection import detect_rsp_peaks_unwrap
from pylife.detection import detect_peaks_r_unwrap

### new 
from pylife.detect_ECG_Peaks import unwrap_peak_ampl, getPeaks_unwrap


from pylife.detection import detect_valid_peaks_breath_unwrap
# from pylife.detection import detect_qrs_brassiere
# from pylife.detection import detect_qrs_brassiere_unwrap
# from pylife.detection import detect_breathing_EDR_unwrap
# from pylife.detection import detect_breathing_EDR
from pylife.activity_measurement import compute_n_steps_unwrap
from pylife.activity_measurement import compute_n_steps
from pylife.activity_measurement import signal_magnetude_area_unwrap
from pylife.activity_measurement import signal_magnetude_area
#from pylife.activity_measurement import compute_activity_level
from pylife.activity_measurement import compute_activity_level_unwrap
from pylife.show_functions import poincare_plot
from pylife.show_functions import ppm_distplot
# from pylife.time_functions import get_time_intervals 
# from pylife.time_functions import get_time_intervals_unwrap
# from pylife.time_functions import datetime_np2str
# from pylife.time_functions import datetime_str2np

import time

from pylife.env import get_env
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import matplotlib.pyplot as plt
    from pylife.show_functions import show_indicators
    import matplotlib.patches as mpatches
# --- Add imports for PROD and DEV env


class Siglife():
    """ Mother class for all signal types """

    def __init__(self, params):
        """ Constructor

        Parameters
        ----------------
         params: dictionary containing signal information
            times: signal times
            values: signal values
            stats: signal stats
            durations: signal and disconnection durations
            fs: signal sampling frequency
            is_empty: flag for empty signal (boolean)

        """

        # CIA: Check / Init Assign parameters
        self.check(params)
        self.init()
        self.assign(params)

    def check(self, params):
        # Missing parameter
        assert 'times'      in params.keys(), "times parameter is missing"
        assert 'sig'        in params.keys(), "sig parameter is missing"
        assert 'fs'         in params.keys(), "fs parameter is missing"
        # assert 'fw_version' in params.keys(), "fw_version parameter is missing"
        assert len(params['sig']) == len(params['times']),\
            "sig and times should have the same length"

    def init(self):
        """ Initialize parameters """

        self.is_empty_              = False
        self.is_fw_version_ok_      = False
        self.n_segments_            = None
        self.time_start_            = None
        self.time_stop_             = None
        self.flag_filt_             = False
        self.flag_clean_            = False
        self.flag_analyze_          = False
        self.clean_step_            = None
        self.analyze_on_sig_        = None
        self.is_wrapped_            = None
        self.flag_error_            = False
        self.app_version_           = None
        self.diagwear_name_         = None
        self.error_names_           = []
        self.log_                   = []
        self.still_times_           = []
        self.move_times_            = []
        self.indicators_worn_       = []
        self.is_worn_               = []
        self.fw_version_            = -1
        self.card_version_          = -1
        self.hard_version_          = -1

        self.sig_                   = []
        self.times_                 = []
        self.fs_                    = None
        self.sig_reshape_           = []
        self.times_reshape_         = []
        self.sig_filt_              = []
        self.times_filt_            = []
        self.sig_clean_             = []
        self.times_clean_           = []
        self.indicators_clean_      = []
        self.indicators_clean_bis_  = []
        self.sig_clean_2_           = []
        self.times_clean_2_         = []
        self.indicators_clean_2_    = []
        self.indicators_clean_1_2_  = []
        self.sig_clean_3_           = []
        self.times_clean_3_         = []
        self.indicators_clean_3_    = []
        self.sig_clean_4_           = []
        self.times_clean_4_         = []
        self.indicators_clean_4_    = []
        self.sig_clean_5_           = []
        self.times_clean_5_         = []
        self.indicators_clean_5_    = []
        self.indicators_frequency_  = []
        
        self.sig_fromto_            = None
        self.times_fromto_          = None
        self.first_timestamps_      = None
        self.last_timestamps_       = None

        self.sig_durations_         = None            
        self.sig_duration_          = None
        self.sig_duration_min_      = None
        self.sig_duration_max_      = None
        self.sig_duration_median_   = None
        self.sig_duration_iqr_      = None
        self.sig_duration_mean_     = None
        self.sig_duration_std_      = None
        self.n_samples_per_sig_     = None
        self.n_samples_per_sig_max_ = None
        
        self.disconnection_durations_           = None
        self.disconnection_duration_            = None
        self.disconnection_duration_min_        = None
        self.disconnection_duration_max_        = None
        self.disconnection_duration_median_     = None
        self.disconnection_duration_iqr_        = None
        self.disconnection_duration_mean_       = None
        self.disconnection_duration_std_        = None
        self.disconnections_number_             = 0
        self.disconnection_percentage_          = None
        self.disconnection_sample_percentage_   = None
        self.disconnection_times_start_         = None
        self.disconnection_times_stop_          = None
        
        self.sig_clean_n_sample_            = None
        self.sig_clean_percentage_          = None
        self.sig_clean_duration_min_        = None
        self.sig_clean_duration_max_        = None
        self.sig_clean_duration_median_     = None
        self.sig_clean_duration_iqr_        = None
        self.sig_clean_duration_mean_       = None
        self.sig_clean_duration_std_        = None
        self.sig_clean_duration_            = None
        self.sig_clean_n_segments_          = None
        
        self.sig_not_clean_n_sample_        = None
        self.sig_not_clean_percentage_      = None
        self.sig_not_clean_duration_min_    = None
        self.sig_not_clean_duration_max_    = None
        self.sig_not_clean_duration_median_ = None
        self.sig_not_clean_duration_iqr_    = None
        self.sig_not_clean_duration_mean_   = None
        self.sig_not_clean_duration_std_    = None
        self.sig_not_clean_duration_        = None
        
        self.plot_colors_ = None

    def assign(self, params):
        """ Load parameters
        Parameters
        ----------------
         params: dictionary containing signal information
            times: signal times
            values: signal values
            stats: signal stats
            durations: signal and disconnection durations
            fs: signal sampling frequency
            is_empty: flag for empty signal (boolean)
        """

        if len(params['sig']) == 0:
            self.is_empty_      = True
            self.flag_error_    = True
            self.error_names_   = ['SIGNAL_EMPTY']
        else:
            self.sig_           = set_list_of_list(params['sig'])
            self.times_         = set_list_of_list(params['times'])
            self.fs_            = params['fs']
            if 'fw_version' in params.keys():
                if type(params['fw_version']) == list or type(params['fw_version']) == np.ndarray:
                    hard_versions   = []
                    card_versions   = []
                    for fw_version in params['fw_version']:
                        res = fw_version_test(fw_version)
                        if not res['error']:
                            hard_versions.append(res['hard_version'])
                            card_versions.append(res['card_version'])
                    assert len(hard_versions) < 2, "fw version should be unique"
                    
                    if len(hard_versions) > 0:
                        self.fw_version_        = params['fw_version'][0]
                        self.hard_version_      = hard_versions[0]
                        self.card_version_      = card_versions[0]
                        self.is_fw_version_ok_  = True
            # else:
            #     fw_version = params['fw_version']
            #     res = fw_version_test(fw_version)
            #     print(res)
            #     if not res['error']:
            #         self.fw_version_        = params['fw_version']
            #         self.hard_version_      = res['hard_version']
            #         self.card_version_      = res['card_version']
            #         self.is_fw_version_ok_  = True
                    
            # if res['error']:
            #     raise NameError('Firmware version format is not correct')
                    
        if 'device_model' in params.keys():
            self.device_model_ = params['device_model']
        else:
            self.device_model_ = 't-shirt'
            
        if 'app_version' in params.keys():
            self.app_version_ = params['app_version']
            
        if 'diagwear_name' in params.keys():
            self.diagwear_name_ = params['diagwear_name']

        self.set_is_wrapped()
        self.set_durations_info()

        if DEV:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            self.plot_colors_ = prop_cycle.by_key()['color']

    def select_on_sig(self, on_sig, on_indicator=None):
        """ Select a given value type of signal

        Parameters
        ----------------
        on_sig: value type of signal

        Returns
        ----------------
        Selected parameters of: times, sig, indicators

        """
        
        if on_sig == 'raw':
            times   = self.times_
            sig     = self.sig_
        elif on_sig == 'var':
            times   = self.times_
            sig     = self.sig_var_
        elif on_sig == 'median':
            times   = self.times_median_
            sig     = self.sig_median_
        elif on_sig == 'var_median':
            times   = self.times_median_
            sig     = self.sig_var_median_
        elif on_sig == 'reshape':
            times   = self.times_reshape_
            sig     = self.sig_reshape_
        elif on_sig == 'filt':
            times   = self.times_filt_
            sig     = self.sig_filt_
        elif on_sig == 'clean_1' or on_sig == 'clean' or on_sig == 'clean_bis':
            times   = self.times_clean_
            sig     = self.sig_clean_
        elif on_sig == 'clean_2':
            times   = self.times_clean_2_
            sig     = self.sig_clean_2_
        elif on_sig == 'clean_3':
            times   = self.times_clean_3_
            sig     = self.sig_clean_3_
        elif on_sig == 'clean_4':
            times   = self.times_clean_4_
            sig     = self.sig_clean_4_
        elif on_sig == 'clean_5':
            times   = self.times_clean_5_
            sig     = self.sig_clean_5_
        elif on_sig == 'clean_var':
            times   = self.times_clean_
            sig     = self.sig_clean_var_
        elif on_sig == 'clean_median':
            times   = self.times_clean_median_
            sig     = self.sig_clean_median_
        elif on_sig == 'clean_var_median':
            times   = self.times_clean_median_
            sig     = self.sig_clean_var_median_
        else:
            raise NameError('on_sig is not correct')
            
        indicators_clean = None
        if on_sig == 'raw' or on_sig == 'filt':
            if self.clean_step_ == 1:
                indicators_clean = self.indicators_clean_
            elif self.clean_step_ == 2:
                indicators_clean = self.indicators_clean_2_
            elif self.clean_step_ == 3:
                indicators_clean = self.indicators_clean_3_
            elif self.clean_step_ == 4:
                indicators_clean = self.indicators_clean_4_
            elif self.clean_step_ == 5:
                indicators_clean = self.indicators_clean_5_
        elif on_sig == 'clean_1' or on_sig == 'clean':
            indicators_clean = self.indicators_clean_1_2_
            
        if on_indicator==1:
            indicators_clean = self.indicators_clean_
        elif on_indicator == 'bis':
                indicators_clean = self.indicators_clean_bis_
        elif on_indicator==2:
            indicators_clean = self.indicators_clean_2_
        elif on_indicator==3:
            indicators_clean = self.indicators_clean_3_
        elif on_indicator==4:
            indicators_clean = self.indicators_clean_4_
        elif on_indicator==5:
            indicators_clean = self.indicators_clean_5_

        return times, sig, indicators_clean

    def select_on_times(self, times, values, from_time, to_time):    
        new_times, new_values, _ = remove_timestamps_unwrap(times, values,
                                                            from_time,
                                                            to_time)
        return new_times, new_values

    def set_durations_info(self):

        if self.is_empty_:
            return

        sig     = self.sig_
        times   = self.times_
        fs      = self.fs_
        info = get_durations_info(times, sig, fs)
        sig_durations           = info['sig_durations'] 
        n_samples_per_sig       = info['n_samples_per_sig'] 
        first_timestamps        = info['first_timestamps'] 
        last_timestamps         = info['last_timestamps'] 
        disconnection_durations = info['disconnection_durations'] 
            
        time_start  = self.times_[0][0]
        time_stop   = self.times_[-1][-1]
        self.time_start_                        = time_start
        self.time_stop_                         = time_stop
        self.sig_durations_                     = sig_durations
        self.n_samples_per_sig_                 = n_samples_per_sig
        self.first_timestamps_                  = first_timestamps
        self.last_timestamps_                   = last_timestamps
        self.disconnection_durations_           = disconnection_durations        
        self.disconnection_number_              = len(first_timestamps)-1
        self.disconnection_sample_percentage_   = np.sum(disconnection_durations)\
            /(np.sum(n_samples_per_sig)/fs)*100
        if time_start != time_stop:
            self.disconnection_percentage_      = np.sum(disconnection_durations)\
                /((time_stop - time_start)/np.timedelta64(1, 's'))*100
        else:
            self.disconnection_percentage_      = 0
        self.disconnection_times_start_         = last_timestamps[:-1]
        self.disconnection_times_stop_          = first_timestamps[1:]
        
        self.n_segments_                        = len(self.sig_)
        self.n_samples_per_sig_max_             = np.max(self.n_samples_per_sig_)

        self.disconnection_duration_            = np.sum(self.disconnection_durations_)
        self.disconnection_duration_min_        = np.min(self.disconnection_durations_)
        self.disconnection_duration_max_        = np.max(self.disconnection_durations_)
        self.disconnection_duration_median_     = np.median(self.disconnection_durations_)
        self.disconnection_duration_iqr_        = np.percentile(self.disconnection_durations_, 75) - np.percentile(self.disconnection_durations_, 25)
        self.disconnection_duration_mean_       = np.mean(self.disconnection_durations_)
        self.disconnection_duration_std_        = np.std(self.disconnection_durations_)
        
        self.sig_duration_                      = self.sig_durations_.sum()
        self.sig_duration_min_                  = np.min(self.sig_durations_)
        self.sig_duration_max_                  = np.max(self.sig_durations_)
        self.sig_duration_median_               = np.median(self.sig_durations_)
        self.sig_duration_iqr_                  = np.percentile(self.sig_durations_, 75) - np.percentile(self.sig_durations_, 25)
        self.sig_duration_mean_                 = np.mean(self.sig_durations_)
        self.sig_duration_std_                  = np.std(self.sig_durations_)
        
    def get_durations_info_fromto(self, from_time, to_time):
        
        output = {}
        if self.is_empty_:
            return output
        times = self.times_
        sig = self.sig_
        fs = self.fs_
        times, sig = self.select_on_times(times, sig, from_time, to_time)
        if len(times) == 0:
            return output
            
        info = get_durations_info(times, sig, fs)
        sig_durations               = info['sig_durations'] 
        n_samples_per_sig           = info['n_samples_per_sig'] 
        first_timestamps            = info['first_timestamps'] 
        last_timestamps             = info['last_timestamps'] 
        disconnection_durations     = info['disconnection_durations'] 
        
        time_start  = times[0][0]
        time_stop   = times[-1][-1]
        
        output['time_start']                        = time_start
        output['time_stop']                         = time_stop
        output['sig_durations']                     = sig_durations
        output['n_samples_per_sig']                 = n_samples_per_sig
        output['first_timestamps']                  = first_timestamps
        output['last_timestamps']                   = last_timestamps
        output['disconnection_durations']           = disconnection_durations        
        output['disconnection_number']              = len(first_timestamps)-1
        output['disconnection_sample_percentage']   = np.sum(disconnection_durations)\
            /(np.sum(n_samples_per_sig)/fs)*100
        
        if time_start != time_stop:
            output['disconnection_percentage']      = np.sum(disconnection_durations)\
                /((time_stop - time_start)/np.timedelta64(1, 's'))*100
        else:
            output['disconnection_percentage']      = 0
        output['disconnection_times_start']         = last_timestamps[:-1]
        output['disconnection_times_stop']          = first_timestamps[1:]
        output['n_segments']                        = len(sig)
        output['n_samples_per_sig_max']             = np.max(n_samples_per_sig)
        output['disconnection_duration']            = np.sum(disconnection_durations)            
        output['disconnection_duration_min']        = np.min(disconnection_durations)
        output['disconnection_duration_max']        = np.max(disconnection_durations)
        output['disconnection_duration_median']     = np.median(disconnection_durations)
        output['disconnection_duration_iqr']        = np.percentile(disconnection_durations, 75) - np.percentile(disconnection_durations, 25)
        output['disconnection_duration_mean']       = np.mean(disconnection_durations)
        output['disconnection_duration_std']        = np.std(disconnection_durations)
        
        output['sig_duration']                      = sig_durations.sum()
        output['sig_duration_min']                  = np.min(sig_durations)
        output['sig_duration_max']                  = np.max(sig_durations)
        output['sig_duration_median']               = np.median(sig_durations)
        output['sig_duration_iqr']                  = np.percentile(sig_durations, 75) - np.percentile(sig_durations, 25)
        output['sig_duration_mean']                 = np.mean(sig_durations)
        output['sig_duration_std']                  = np.std(sig_durations)
        
        return output

    def set_is_wrapped(self):
        """ Set is_wrapped: Define if input signal is a matrix """
        if self.is_empty_:
            return
        self.is_wrapped_ = is_list_of_list(self.sig_)

    def set_stats_clean_sig(self):
        """ Set stats for clean signal:
            sig_clean_n_sample_per_segment : number of cleaned sample
            sig_clean_percentage_per_segment : percentage of cleaned sample
        """
        if self.is_empty_:
            return
        
        if self.clean_step_ == 1:
            if len(self.indicators_clean_bis_) > 0:
                indicators_clean = self.indicators_clean_bis_
            else:
                indicators_clean = self.indicators_clean_
            
        elif self.clean_step_ == 2:
            indicators_clean = self.indicators_clean_2_
        elif self.clean_step_ == 3:
            indicators_clean = self.indicators_clean_3_
        elif self.clean_step_ == 4:
            indicators_clean = self.indicators_clean_4_
        elif self.clean_step_ == 5:
            indicators_clean = self.indicators_clean_5_
        else:
            return
        fs = self.fs_

        info = get_stats_clean_sig(indicators_clean, fs)
        
        sig_clean_n_sample          = info['n_sample']
        sig_clean_percentage        = info['percentage']
        sig_clean_duration_min      = info['duration_min']
        sig_clean_duration_max      = info['duration_max']
        sig_clean_duration_median   = info['duration_median']
        sig_clean_duration_iqr      = info['duration_iqr']
        sig_clean_duration_mean     = info['duration_mean']
        sig_clean_duration_std      = info['duration_std']
        sig_clean_duration          = info['duration'] 
        
        if self.alias_ == 'ecg':
            self.sig_clean_n_segments_ = count_usable_segments(indicators_clean, 
                                                               fs, 
                                                               window_smooth_indicators=1, 
                                                               duration=30)
        elif self.alias_ == 'breath_1' or self.alias_ == 'breath_2':
            self.sig_clean_n_segments_ = count_usable_segments(indicators_clean,
                                                               fs, 
                                                               window_smooth_indicators=4, 
                                                               duration=60)
            
        self.sig_clean_n_sample_            = sig_clean_n_sample
        self.sig_clean_percentage_          = sig_clean_percentage
        self.sig_clean_duration_min_        = sig_clean_duration_min
        self.sig_clean_duration_max_        = sig_clean_duration_max
        self.sig_clean_duration_median_     = sig_clean_duration_median
        self.sig_clean_duration_iqr_        = sig_clean_duration_iqr
        self.sig_clean_duration_mean_       = sig_clean_duration_mean
        self.sig_clean_duration_std_        = sig_clean_duration_std
        self.sig_clean_duration_            = sig_clean_duration
    
    def set_stats_not_clean_sig(self):
        """ Set stats for clean signal:
            sig_clean_n_sample_per_segment : number of cleaned sample
            sig_clean_percentage_per_segment : percentage of cleaned sample
        """
        if self.is_empty_:
            return
        
        if self.clean_step_ == 1:
            if len(self.indicators_clean_bis_) > 0:
                indicators_clean = self.indicators_clean_bis_
            else:
                indicators_clean = self.indicators_clean_
        elif self.clean_step_ == 2:
            indicators_clean = self.indicators_clean_2_
        elif self.clean_step_ == 3:
            indicators_clean = self.indicators_clean_3_
        elif self.clean_step_ == 4:
            indicators_clean = self.indicators_clean_4_
        elif self.clean_step_ == 5:
            indicators_clean = self.indicators_clean_5_
        else:
            return
        fs = self.fs_

        info = get_stats_not_clean_sig(indicators_clean, fs)
        sig_not_clean_n_sample          = info['n_sample']
        sig_not_clean_percentage        = info['percentage']
        sig_not_clean_duration_min      = info['duration_min']
        sig_not_clean_duration_max      = info['duration_max']
        sig_not_clean_duration_median   = info['duration_median']
        sig_not_clean_duration_iqr      = info['duration_iqr']
        sig_not_clean_duration_mean     = info['duration_mean']
        sig_not_clean_duration_std      = info['duration_std']
        sig_not_clean_duration          = info['duration'] 
    
        self.sig_not_clean_n_sample_            = sig_not_clean_n_sample
        self.sig_not_clean_percentage_          = sig_not_clean_percentage
        self.sig_not_clean_duration_min_        = sig_not_clean_duration_min
        self.sig_not_clean_duration_max_        = sig_not_clean_duration_max
        self.sig_not_clean_duration_median_     = sig_not_clean_duration_median
        self.sig_not_clean_duration_iqr_        = sig_not_clean_duration_iqr
        self.sig_not_clean_duration_mean_       = sig_not_clean_duration_mean
        self.sig_not_clean_duration_std_        = sig_not_clean_duration_std
        self.sig_not_clean_duration_            = sig_not_clean_duration
        
    def get_stats_clean_sig_fromto(self, from_time, to_time):

        if self.is_empty_:
            return
        times_ini = self.times_
        fs = self.fs_        
        if self.clean_step_ == 1:
            if len(self.indicators_clean_bis_) > 0:
                indicators_clean = self.indicators_clean_bis_
            else:
                indicators_clean = self.indicators_clean_
        elif self.clean_step_ == 2:
            indicators_clean = self.indicators_clean_2_
        elif self.clean_step_ == 3:
            indicators_clean = self.indicators_clean_3_
        elif self.clean_step_ == 4:
            indicators_clean = self.indicators_clean_4_
        elif self.clean_step_ == 5:
            indicators_clean = self.indicators_clean_5_
        else:
            return
            
        times, indicators_clean = self.select_on_times(times_ini, 
                                                       indicators_clean,
                                                       from_time, to_time)
        
        output = get_stats_clean_sig(indicators_clean, fs)
        
        output['n_segments'] = None
        if self.alias_ == 'ecg':
            output['n_segments'] = count_usable_segments(indicators_clean, 
                                                         fs, 
                                                         window_smooth_indicators=1, 
                                                         duration=30)
        elif self.alias_ == 'breath_1' or self.alias_ == 'breath_2':
            output['n_segments'] = count_usable_segments(indicators_clean, 
                                                         fs, 
                                                         window_smooth_indicators=4, 
                                                         duration=60)

        return output
    
    def get_stats_not_clean_sig_fromto(self, from_time, to_time):

        if self.is_empty_:
            return
        times_ini = self.times_
        fs = self.fs_        
        if self.clean_step_ == 1:
            if len(self.indicators_clean_bis_) > 0:
                indicators_clean = self.indicators_clean_bis_
            else:
                indicators_clean = self.indicators_clean_
        elif self.clean_step_ == 2:
            indicators_clean = self.indicators_clean_2_
        elif self.clean_step_ == 3:
            indicators_clean = self.indicators_clean_3_
        elif self.clean_step_ == 4:
            indicators_clean = self.indicators_clean_4_
        elif self.clean_step_ == 5:
            indicators_clean = self.indicators_clean_5_
        else:
            return
            
        times, indicators_clean = self.select_on_times(times_ini, 
                                                       indicators_clean,
                                                       from_time, to_time)
        

        output = get_stats_not_clean_sig(indicators_clean, fs)

        return output

    # def set_rate_pm_info(self, on_sig='clean'):

    #     times, sig, indicators_clean = self.select_on_sig(on_sig)
    #     rate_pm_times       = self.rate_pm_times_
    #     rate_pm_times_start = []
    #     rate_pm_times_stop  = []
    
    #     for id_seg in range(len(sig)):
    #         start = []
    #         stop = []
    #         bmp_times_seg = rate_pm_times[id_seg]
    #         for rate_pm_time in bmp_times_seg:
    #             if len(rate_pm_time) > 0:
    #                 start.append(rate_pm_time[0])
    #                 stop.append(rate_pm_time[1])
    #         rate_pm_times_start.append(start)
    #         rate_pm_times_stop.append(stop)

    #     self.rate_pm_times_start_   = rate_pm_times_start
    #     self.rate_pm_times_stop_    = rate_pm_times_stop

    def show(self, id_seg=None, on_sig='raw', on_indicator=None, 
             show_indicators=None, from_time=None, to_time=None, color=None, 
             center=False):
        """ Display signal:
            If segment id is defined, segment signal is displayed
            else, the whole

        Parameters
        ----------------
        id_seg:             id of signal's segment
        on_sig:         value type of signal
        indicators:    quality indicators_clean of signal (boolean)

        """

        if on_sig.lower() == 'filt' and not self.flag_filt_:
            print('Signal filt does not exist. Run filt() or clean() first')
            return

        if on_sig.lower() == 'clean' and not self.flag_clean_:
            print('Signal clean does not exist. Run clean() method first')
            return

        times_init, sig_init, indicators_clean_init = self.select_on_sig(on_sig=on_sig,
                                                                   on_indicator=on_indicator)
        
        if from_time is not None or to_time is not None:
            times, sig = self.select_on_times(times_init, sig_init, from_time,
                                              to_time)

            if show_indicators == 'clean':
                _, indicators_clean = self.select_on_times(times_init,
                                                           indicators_clean_init,
                                                           from_time, to_time)
            if show_indicators == 'worn':
                _, indicators_worn = self.select_on_times(times_init,
                                                           self.indicators_worn_,
                                                           from_time, to_time)
        else:
            times = times_init
            sig = sig_init
            if show_indicators == 'clean':
                indicators_clean = indicators_clean_init
            if show_indicators == 'worn':
                indicators_worn = self.indicators_worn_

        if show_indicators is not None:
            if not self.flag_clean_:
                print('Run "clean" method to show still times. Clean may be impossible if signals have not the same length (still/sleep detection)') 
                return
        if show_indicators == 'clean':
            indicators = indicators_clean
        elif show_indicators == 'worn':
            indicators = indicators_worn
        else:
            indicators = None

        if center:
            for i in range(len(sig)):
                sig[i] = sig[i] - np.mean(sig[i])

        if id_seg is not None:
            self.show_segment(times, sig, id_seg, on_sig, indicators=indicators,
                              ccolor=color)
        else:
            self.show_unwrap(times, sig, on_sig, indicators=indicators, color=color)
            
        if show_indicators in ['clean', 'worn']:
            plt.legend(fontsize=14).get_texts()[1].set_text('Indicators ' + show_indicators)

    def show_segment(self, times, sig, id_seg, on_sig, indicators=None, color=None):
        """ Display a segment of the signal

        Parameters
        ----------------
        id_seg:             id of signal's segment
        on_sig:             value type of signal
        indicators_clean:   quality indicators_clean of signal (boolean)

        """
        if not DEV:
            return
        
        if not is_list_of_list(sig):
            raise NameError('show_segment is only applicable on matrix signal')

        seg = sig[id_seg]
        times_seg = times[id_seg]

        if indicators is not None:
            indicators_seg = indicators[id_seg]
            show_indicators(times_seg, seg, indicators=indicators_seg)

        else:
            if color is not None:
                g, = plt.plot(times_seg, seg, color=color)
            else:
                g, = plt.plot(times_seg, seg)
                
        if on_sig[:5] == 'clean':
            title = self.name_ + ' ' + 'clean'
        else:
            title = self.name_ + ' ' + on_sig
        if on_sig == 'var':
            title = self.name_ + ' ' + 'variation'
        if on_sig == 'var_median':
            title = self.name_ + ' ' + 'variation median'
        if on_sig == 'clean_var_median':
            title = self.name_ + ' ' + 'clean variation median'
        
        self.define_plot_settings(title)

    def show_single_segment(self, times, sig, on_sig, indicators=None, color=None):
        """ Display a segment of the signal

        Parameters
        ----------------
        on_sig:         value type of signal
        indicators:    quality indicators_clean of signal (boolean)

        """
        if not DEV:
            return
        
        if is_list_of_list(sig):
            raise NameError('show_segment is only applicable on matrix signal')

        if indicators is not None:
            show_indicators(times, sig, indicators=indicators, color=color)
        else:
            if color is not None:
                g, = plt.plot(times, sig, color=color)
            else:
                g, = plt.plot(times, sig)

        if on_sig[:5] == 'clean':
            title = self.name_ + ' ' + 'clean'
        else:
            title = self.name_ + ' ' + on_sig
        if on_sig == 'var':
            title = self.name_ + ' ' + 'variation'
        if on_sig == 'var_median':
            title = self.name_ + ' ' + 'variation median'
        if on_sig == 'clean_var_median':
            title = self.name_ + ' ' + 'clean variation median'
            
        self.define_plot_settings(title)

    def show_unwrap(self, times, sig, on_sig, indicators=None, color=None):
        """ Display the whole signal with time discontinuities

        Parameters
        ----------------
        on_sig:         value type of signal
        indicators:    quality indicators_clean of signal (boolean)
        """
        if not DEV:
            return
        
        if not self.is_wrapped_:
            raise NameError('show_unwrap_colors is only applicable on matrix signal')

        for id_seg, seg in enumerate(sig):
            times_seg = times[id_seg]
            if indicators is not None:
                if len(indicators) > 0:
                    indicators_seg = indicators[id_seg]
                    show_indicators(times_seg, seg, indicators=indicators_seg, color=color)
            else:
                if color is not None:
                    try:
                        g, = plt.plot(times_seg, seg, color=color)
                    except:
                        print()
                else:
                    g, = plt.plot(times_seg, seg)
                    
        if on_sig[:5] == 'clean':
            title = self.name_ + ' ' + 'clean'
        else:
            title = self.name_ + ' ' + on_sig
        if on_sig == 'var':
            title = self.name_ + ' ' + 'variation'
        if on_sig == 'var_median':
            title = self.name_ + ' ' + 'variation median'
        if on_sig == 'clean_var_median':
            title = self.name_ + ' ' + 'clean variation median'
            
        self.define_plot_settings(title)

    def define_plot_settings(self, title):
        """ Define plot labels """
        
        if not DEV:
            return
        
        fontsize=14
        plt.title(title, fontsize=fontsize)
        plt.xlabel('Time', fontsize=fontsize)
        plt.xticks(fontsize=8)
        plt.ylabel('Amplitude', fontsize=fontsize)

    def time_shift(self, offset, time_format='h'):

        if self.is_empty_:
            return

        times = self.times_
        new_times = []    
        for time in times:
            new_times.append(time + np.timedelta64(offset, time_format))

        self.times_ = new_times
        self.set_durations_info()

        if len(self.times_clean_) > 0:
            times = self.times_clean_
            new_times = []
        
            for time in times:
                new_times.append(time + np.timedelta64(offset,
                                                       time_format))
            self.times_clean_ = new_times

            if self.alias_ in ['ecg', 'breath_1', 'breath_2'] and\
                    len(self.peaks_times_) > 0:
                peaks_times = self.peaks_times_
                new_peaks_times = []
                
                for peak_time in peaks_times:
                    new_peaks_times.append(peak_time
                                           + np.timedelta64(offset,
                                                            time_format))
                self.peaks_times_ = new_peaks_times

        if self.alias_ in ['accx', 'accy', 'accz']:
            if len(self.steps_times_start_) > 0:
                times = self.steps_times_start_
                new_times = []
                
                for time in times:
                    new_times.append(time + np.timedelta64(offset,
                                                           time_format))
                self.steps_times_start_ = new_times

        elif self.alias_ in ['breath_1', 'breath_2', 'ecg']:
            if len(self.rate_pm_times_start_) > 0:
                times = self.rate_pm_times_start_
                new_times = []                
                for time in times:
                    new_times.append(time + np.timedelta64(offset,
                                                           time_format))
                self.rate_pm_times_start_ = new_times

    def show_disconnections(self):
        n_disconnections        = len(self.first_timestamps_)-1
        duration_disconnections = np.sum(self.disconnection_durations_)
        print(n_disconnections, 'disconnections', duration_disconnections, 'seconds')
        for i in range(n_disconnections):
            print('From',
                  # str(al.accx.last_timestamps_[i])[:10],
                  str(self.last_timestamps_[i])[11:19],
                  'to',
                  # str(al.accx.first_timestamps_[i+1])[:10],
                  str(self.first_timestamps_[i+1])[11:19],
                  '~',
                  int(self.disconnection_durations_[i]), 
                  'seconds')
            
    def show_random_segments(self, window, on_sig='clean', n_seg=3, fontsize=14, path_save=None):
        
        if not DEV:
            return
        
        times, sig, _ = self.select_on_sig(on_sig)
        fs = self.fs_
        
        for i in range(n_seg):
            if len(times) > 0:
                iseg = randint(0, len(sig)-1)
                seg = sig[iseg][:fs*window]
                plt.figure()
                
                if on_sig[:5] == 'clean':
                    title = self.name_ + ' ' + 'clean'
                else:
                    title = self.name_ + ' ' + on_sig.upper()
                if on_sig == 'var':
                    title = self.name_ + ' ' + 'variation'
                if on_sig == 'var_median':
                    title = self.name_ + ' ' + 'variation median'
                if on_sig == 'clean_var_median':
                    title = self.name_ + ' ' + 'clean variation median'
                
                plt.title(title, fontsize=fontsize)
                plt.plot(seg)
                start = times[iseg][0]
                start = str(np.datetime64(start, 's')).replace('T', ' ')
                start = start[11:]
                
                stop = times[iseg][len(seg)-1]
                stop = str(np.datetime64(stop, 's')).replace('T', ' ')
                stop = stop[11:]
                plt.xlabel('from ' + start + ' to ' + stop)
            else:
                plt.figure()
                if on_sig[:5] == 'clean':
                    title = self.name_ + ' ' + 'clean'
                else:
                    title = self.name_ + ' ' + on_sig
                if on_sig == 'var':
                    title = self.name_ + ' ' + 'variation'
                if on_sig == 'var_median':
                    title = self.name_ + ' ' + 'variation median'
                if on_sig == 'clean_var_median':
                    title = self.name_ + ' ' + 'clean variation median'
                plt.title(title, fontsize=fontsize)
            plt.xticks([])
            if path_save is not None:
                plt.savefig(path_save + self.alias_ + '_random_seg' + str(i) + '.png')

###############################################################################
class SiglifePeriodic(Siglife):
    
    def show_peaks(self, id_seg=None, from_time=None, to_time=None,
                   color=None, center=False, on_sig='filt'):

        if 'clean' in on_sig and not self.flag_analyze_:
            raise NameError('Run analyze to show peaks')
        show_clean = False
        
        times_init, sig_init, indicators_init = self.select_on_sig(on_sig)
        
        if on_sig[:5] == 'clean':
            peaks_times = self.peaks_times_clean_
        else:
            peaks_times = self.peaks_times_

        if from_time is not None or to_time is not None:
            times, sig = self.select_on_times(times_init, sig_init,
                                              from_time, to_time)

        else:
            times = times_init
            sig = sig_init
            del sig_init, times_init

        if not show_clean:
            indicators_clean = None
    
        if center:
            for i in range(len(sig)):
                sig[i] = sig[i] - np.mean(sig[i])
        if id_seg is not None:
            self.show_peaks_segment(times, sig, id_seg, peaks_times, on_sig,
                                    indicators_clean=indicators_clean, color=color)
        else:
            self.show_peaks_unwrap(times, sig, peaks_times, on_sig,
                                   indicators=indicators_clean, color=color)

    def show_peaks_segment(self, times, sig, id_seg, peaks_times, on_sig,
                           indicators=None, color=None):
        
        if not DEV:
            return
        
        times = times[id_seg]
        sig = sig[id_seg]

        self.show_single_segment(times, sig, on_sig, color=color)
        peaks = get_peaks_from_peaks_times(times, peaks_times)
        if len(peaks) > 0:
            times = np.array(times)
            sig = np.array(sig)
            plt.plot(times[peaks], sig[peaks], 'ok')

    def show_peaks_unwrap(self, times, sig, peaks_times, on_sig, indicators=None,
                          color=None):
        
        if not DEV:
            return
        
        if not is_list_of_list(sig):
            raise NameError('show_unwrap_colors is only applicable on matrix signal')

        for id_seg, seg in enumerate(sig):
            times_seg = times[id_seg]
            self.show_single_segment(times_seg, seg, on_sig, color=color)
            peaks = get_peaks_from_peaks_times(times_seg, peaks_times)
            if len(peaks) > 0:
                plt.plot(np.array(times_seg)[peaks], np.array(seg)[peaks], 'ok')

    def show_poincare(self, from_time=None, to_time=None):

        if not self.flag_analyze_:
            raise NameError('Run analyze to show peaks')

        on_sig = 'clean'
        times_init, sig_init, _ = self.select_on_sig(on_sig)
        fs = self.fs_
        peaks_times = self.peaks_times_
        if from_time is not None or to_time is not None:
            times, _ = self.select_on_times(times_init, sig_init,
                                            from_time, to_time)
        else:
            times = times_init

        for id_seg, seg in enumerate(times):
            times_seg = times[id_seg]
            peaks = get_peaks_from_peaks_times(times_seg, peaks_times)

        rr_intervals = pos_2_time_interval_unwrap(fs, peaks)
        poincare_plot(rr_intervals)

    def show_rate_pm(self, from_time=None, to_time=None):
        """ Show rate_pm distribution """

        if self.flag_analyze_:
            on_sig = self.analyze_on_sig_
            times, sig, indicators_clean = self.select_on_sig(on_sig)
            rate_pm = self.rate_pm_
            if from_time is not None or to_time is not None:
                times, rate_pm = self.select_on_times(times, rate_pm,
                                                       from_time, to_time)
            ppm_distplot(unwrap(rate_pm))

        else:
            raise NameError('Run analyze() method before plotting show_heart_rate_rate_pm')

    def define_plot_settings(self, title):
        """ Define plot labels """
        if not DEV:
            return
        
        fontsize=14
        plt.title(title, fontsize=fontsize)
        plt.xlabel('Time', fontsize=fontsize)
        plt.xticks(fontsize=8)
        plt.ylabel('Amplitude', fontsize=fontsize)

    def time_shift(self, offset, time_format='h'):

        if self.is_empty_:
            return

        times = self.times_
        new_times = []    
        for time in times:
            new_times.append(time + np.timedelta64(offset, time_format))

        self.times_ = new_times
        self.set_durations_info()

        if len(self.times_clean_) > 0:
            times = self.times_clean_
            new_times = []
        
            for time in times:
                new_times.append(time + np.timedelta64(offset,
                                                       time_format))
            self.times_clean_ = new_times

            if self.alias_ in ['ecg', 'breath_1', 'breath_2'] and\
                    len(self.peaks_times_) > 0:
                peaks_times = self.peaks_times_
                new_peaks_times = []
                
                for peak_time in peaks_times:
                    new_peaks_times.append(peak_time
                                           + np.timedelta64(offset,
                                                            time_format))
                self.peaks_times_ = new_peaks_times

        if self.alias_ in ['accx', 'accy', 'accz']:
            if len(self.steps_times_start_) > 0:
                times = self.steps_times_start_
                new_times = []
                
                for time in times:
                    new_times.append(time + np.timedelta64(offset,
                                                           time_format))
                self.steps_times_start_ = new_times

        elif self.alias_ in ['breath_1', 'breath_2', 'ecg']:
            if len(self.rate_pm_times_start_) > 0:
                times = self.rate_pm_times_start_
                new_times = []                
                for time in times:
                    new_times.append(time + np.timedelta64(offset,
                                                           time_format))
                self.bpm_times_start_ = new_times

###############################################################################
class Acceleration(Siglife):
    """ Acceleration class """
    name_ = 'Acceleration'
    alias_ = 'acc'

###############################################################################
class Acceleration_x(Acceleration):
    """ Acceleration x class """
    name_ = 'Acceleration x'
    alias_ = 'accx'


###############################################################################
class Acceleration_y(Acceleration):
    """ Acceleration y class """
    name_ = 'Acceleration y'
    alias_ = 'accy'


###############################################################################
class Acceleration_z(Acceleration):
    """ Acceleration z class """
    name_ = 'Acceleration z'
    alias_ = 'accz'


###############################################################################
class Accelerations():
    """ Accelerations class """
    name_ = 'Accelerations'
    alias_ = 'accs'

    def __init__(self, accx, accy, accz):
        """ Constructor

        Parameters
        ---------------
        accx: Acceleration_x object
        accy: Acceleration_y object
        accz: Acceleration_z object

        """
        self.check_params(accx, accy, accz)
        self.init_params()
        self.assign_params()

    def check_params(self, accx, accy, accz):

        self.is_empty_      = False
        self.flag_error_    = False
        self.error_names_   = []
        self.log_           = []
        
        if accx.is_empty_:
            self.flag_error_= True
            self.error_names_.append('ACCX_IS_EMPTY')
        if accy.is_empty_:
            self.flag_error_= True
            self.error_names_.append('ACCY_IS_EMPTY')
        if accz.is_empty_:
            self.flag_error_= True
            self.error_names_.append('ACCZ_IS_EMPTY')
            
        if self.flag_error_:
            self.is_empty_  = True
            return
        self.accx = accx
        self.accy = accy
        self.accz = accz

        
    def init_params(self):
        
        self.sma_                   = []
        self.n_steps_               = []
        self.steps_                 = []
        self.steps_times_start_     = None
        self.steps_times_stop_      = None
        self.mean_activity_         = []
        self.steps_delta_           = []
        self.mean_activity_delta_   = []
        self.times_delta_           = []
        self.activity_level_        = []
        self.mean_activity_level_   = []
    
        if not self.is_empty_:
            if self.accx.flag_analyze_:
                return
            self.set_init_params(self.accx)
            self.set_init_params(self.accy)
            self.set_init_params(self.accz)
        
    def assign_params(self):
        pass

    def filt(self):
        """ Filter signal """

        sig_x   = self.accx.sig_
        sig_y   = self.accy.sig_
        sig_z   = self.accz.sig_
        
      
        t_x     = self.accx.times_
        t_y     = self.accy.times_
        t_z     = self.accz.times_
        
        sig_x_reshape   = []
        sig_y_reshape   = []
        sig_z_reshape   = []
        t_x_reshape     = []
        t_y_reshape     = []
        t_z_reshape     = []
        self.is_length_xyz_equal_ = True
    
        for id_seg in range(min([len(sig_x), len(sig_y), len(sig_z)])):
            if len(sig_x[id_seg]) != len(sig_y[id_seg]) or\
                    len(sig_y[id_seg]) != len(sig_z[id_seg]):
                self.flag_error_= True
                self.error_names_.append('ACCS_LENGTH')
                log = str('\n-------------------------------------------------' +\
                      '\n!!! WARNING !!! Error length accelerations' +\
                      '\nSegment n° ' + str(id_seg) +\
                      '\nTime accx:' + str(t_x[id_seg][0]) +\
                      '\nLength accx:' + str(len(sig_x[id_seg])) +\
                      '\nTime accy:' + str(t_y[id_seg][0]) +\
                      '\nlength accy:' + str(len(sig_y[id_seg])) +\
                      '\nTime accz:' + str(t_z[id_seg][0]) +\
                      '\nlength accz:' + str(len(sig_z[id_seg]))
                      )
                print(log)
                self.log_.append(log)
                
                length_min = np.min([len(sig_x[id_seg]),
                                     len(sig_y[id_seg]),
                                     len(sig_z[id_seg])])
                sig_x_reshape.append(sig_x[id_seg][:length_min])
                sig_y_reshape.append(sig_y[id_seg][:length_min])
                sig_z_reshape.append(sig_z[id_seg][:length_min])
                t_x_reshape.append(t_x[id_seg][:length_min])
                t_y_reshape.append(t_y[id_seg][:length_min])
                t_z_reshape.append(t_z[id_seg][:length_min])
            else:
                sig_x_reshape.append(sig_x[id_seg])
                sig_y_reshape.append(sig_y[id_seg])
                sig_z_reshape.append(sig_z[id_seg])
                t_x_reshape.append(t_x[id_seg])
                t_y_reshape.append(t_y[id_seg])
                t_z_reshape.append(t_z[id_seg])

        sig_x  = np.array(sig_x_reshape)
        sig_y  = np.array(sig_y_reshape)
        sig_z  = np.array(sig_z_reshape)
        
        
        self.accx.sig_reshape_ = np.array(sig_x)
        self.accy.sig_reshape_ = np.array(sig_y)
        self.accz.sig_reshape_ = np.array(sig_z)
        
        self.accx.times_reshape_ = np.array(t_x_reshape)
        self.accy.times_reshape_ = np.array(t_y_reshape)
        self.accz.times_reshape_ = np.array(t_z_reshape)
        
        sig_filt            = filter_acceleration_unwrap(sig_x)
        self.accx.sig_filt_ = set_list_of_list(sig_filt)
        sig_filt            = filter_acceleration_unwrap(sig_y)
        self.accy.sig_filt_ = set_list_of_list(sig_filt)
        sig_filt            = filter_acceleration_unwrap(sig_z)
        self.accz.sig_filt_ = set_list_of_list(sig_filt)
        
        self.accx.times_filt_ = np.array(t_x_reshape)
        self.accy.times_filt_ = np.array(t_y_reshape)
        self.accz.times_filt = np.array(t_z_reshape)

        self.accx.flag_filt_    = True
        self.accy.flag_filt_    = True
        self.accz.flag_filt_    = True

    def clean(self):
        """ Clean accelerations """
      
        if self.is_empty_:
            return

        self.accx.sig_clean_    = self.accx.sig_filt_
        self.accy.sig_clean_    = self.accy.sig_filt_
        self.accz.sig_clean_    = self.accz.sig_filt_
        self.accx.times_clean_  = self.accx.times_filt_
        self.accy.times_clean_  = self.accy.times_filt_
        self.accz.times_clean_  = self.accz.times_filt_

        if self.accx.is_wrapped_:
            indic = []
            for seg in self.accx.sig_:
                indic.append(np.ones(len(seg)))
            self.accx.indicators_clean_ = indic

            indic = []
            for seg in self.accy.sig_:
                indic.append(np.ones(len(seg)))
            self.accy.indicators_clean_ = indic

            indic = []
            for seg in self.accz.sig_:
                indic.append(np.ones(len(seg)))
            self.accz.indicators_clean_ = indic

        else:
            self.accx.indicators_clean_     = np.ones(len(self.accx.sig_))
            self.accy.indicators_clean_     = np.ones(len(self.accy.sig_))
            self.accz.indicators_clean_     = np.ones(len(self.accz.sig_))

        self.accx.clean_step_   = 1
        self.accy.clean_step_   = 1
        self.accz.clean_step_   = 1
        self.accx.set_stats_clean_sig()
        self.accx.set_stats_not_clean_sig()
        self.accy.set_stats_clean_sig()
        self.accy.set_stats_not_clean_sig()
        self.accz.set_stats_clean_sig()
        self.accz.set_stats_not_clean_sig()
        self.accx.flag_clean_   = True
        self.accy.flag_clean_   = True
        self.accz.flag_clean_   = True
        

    def analyze(self, on_sig=None):
        """ Analyze signal """
        
        if self.is_empty_:
            print('----------------------------------------------------------')
            print('Analyze: ' + self.alias_ + ' class is empty')
            return
        
        fs = self.accx.fs_       
        if on_sig is None:
            on_sig = 'reshape'
        times, sig_x, indicators_clean = self.accx.select_on_sig(on_sig=on_sig)
        times, sig_y, indicators_clean = self.accy.select_on_sig(on_sig=on_sig)
        times, sig_z, indicators_clean = self.accz.select_on_sig(on_sig=on_sig)
        
        if len(unwrap(sig_x)) == 0 or len(unwrap(sig_y)) == 0 or len(unwrap(sig_z)) == 0:
            print('--------------------------------------------------------')
            print('At least 1 selected signal of clean accelerations is empty')
            return
        
        n_steps, steps, steps_times_start,\
        steps_times_stop = compute_n_steps_unwrap(times, sig_x,
                                                    sig_y, sig_z,
                                                    fs)
        on_sig = 'clean_' + str(self.accx.clean_step_)
        
        times, sig_x, indicators_clean = self.accx.select_on_sig(on_sig=on_sig)
        times, sig_y, indicators_clean = self.accy.select_on_sig(on_sig=on_sig)
        times, sig_z, indicators_clean = self.accz.select_on_sig(on_sig=on_sig)
        
        if len(unwrap(sig_x)) < 50 or len(unwrap(sig_y)) < 50 or len(unwrap(sig_z)) < 50:
            print('--------------------------------------------------------')
            print('At least 1 selected signal of clean accelerations is too short')
            return
        
        activity_level      = compute_activity_level_unwrap(sig_x, sig_y, sig_z)
        mean_activity_level = np.mean(unwrap(activity_level))
        sma                 = signal_magnetude_area_unwrap(sig_x, sig_y, sig_z, fs)
        mean_activity       = np.mean(unwrap(sma))
    
        n_steps_sum = 0
        for seg_step in range(len(n_steps)):         
            if n_steps[seg_step]:
                # print(n_steps[seg_step])
                n_steps_sum += n_steps[seg_step][-1]
                
        self.n_steps_               = n_steps_sum
        self.steps_                 = steps
        self.steps_times_start_     = steps_times_start
        self.steps_times_stop_      = steps_times_stop
        self.sma_                   = sma
        self.mean_activity_         = mean_activity
        self.activity_level_        = activity_level
        self.mean_activity_level_   = mean_activity_level
        self.analyze_on_sig_        = on_sig
        self.flag_analyze_          = True

        self.set_analyze_results(self.accx)
        self.set_analyze_results(self.accy)
        self.set_analyze_results(self.accz)

    def analyze_v2(self, on_sig=None, nb_delta=5, delta='m'):

        if on_sig is None:
            on_sig = 'raw'
        """ Analyze acc """
        times_x, sig_x, indicators_clean = self.accx.select_on_sig(on_sig)
        times_y, sig_y, indicators_clean = self.accy.select_on_sig(on_sig)
        times_z, sig_z, indicators_clean = self.accz.select_on_sig(on_sig)

        fs = self.accx.fs_

        if len(sig_x) == 0 or len(sig_y) == 0 or len(sig_z) == 0:
            print('--------------------------------------------------------')
            print('At least 1 selected signal of accelerations is empty')
            return

        timelinex   = np.array(unwrap(times_x))
        sig_x       = np.array(unwrap(sig_x))
        sig_y       = np.array(unwrap(sig_y))
        sig_z       = np.array(unwrap(sig_z))
        t           = timelinex[0]

        mean_activity_delta     = []
        steps_delta             = []
        times_delta             = []
        while t < timelinex[-1]:

            sigx_delta  = sig_x[timelinex > t]
            sigy_delta  = sig_y[timelinex > t]
            sigz_delta  = sig_z[timelinex > t]
            tx_delta    = timelinex[timelinex > t]

            sigx_delta  = sigx_delta[tx_delta < t + np.timedelta64(nb_delta, delta)]
            sigy_delta  = sigy_delta[tx_delta < t + np.timedelta64(nb_delta, delta)]
            sigz_delta  = sigz_delta[tx_delta < t + np.timedelta64(nb_delta, delta)]
            tx_delta    = tx_delta[tx_delta < t + np.timedelta64(nb_delta, delta)]
            if tx_delta.tolist():
                 times_delta.append(str(tx_delta[0])[4:16])
                 n_steps, steps_times_start,\
                     n_steps_times_stop = compute_n_steps(tx_delta, sigx_delta,
                                                          sigy_delta,
                                                          sigz_delta, fs)
                 sigx_delta     = filter_acceleration(sigx_delta)
                 sigy_delta     = filter_acceleration(sigy_delta)
                 sigz_delta     = filter_acceleration(sigz_delta)
                 sma            = signal_magnetude_area(sigx_delta, 
                                                        sigy_delta, 
                                                        sigz_delta, 
                                                        fs)
                 mean_activity_delta.append(np.mean(sma))
     
                 if n_steps:
                     steps = n_steps[-1]
                 else:
                     steps = 0
                 steps_delta.append(steps)
            t = t + np.timedelta64(nb_delta, delta)

        self.steps_delta_           = steps_delta
        self.mean_activity_delta_   = mean_activity_delta
        self.times_delta_           = times_delta
        
    def set_init_params(self, obj):
        obj.n_steps_                = 0
        obj.steps_                  = []
        obj.steps_times_start_      = []
        obj.steps_times_stop_       = []
        obj.analyze_on_sig_         = 'clean'
        obj.sma_                    = []
        obj.mean_activity_          = []
        obj.activity_level_         = []
        obj.mean_activity_level_    = []

    def set_analyze_results(self, obj):
        obj.n_steps_                = self.n_steps_
        obj.steps_                  = self.steps_
        obj.steps_times_start_      = self.steps_times_start_
        obj.steps_times_stop_       = self.steps_times_stop_
        obj.sma_                    = self.sma_
        obj.mean_activity_          = self.mean_activity_
        obj.activity_level_         = self.activity_level_
        obj.mean_activity_level_    = self.mean_activity_level_
        obj.analyze_on_sig_         = self.analyze_on_sig_
        obj.flag_analyze_           = self.flag_analyze_


###############################################################################
class Breath(SiglifePeriodic):
    """ Breath class """
    name_ = 'Breath'
    alias_ = 'breath'
    
    def __init__(self, params):
        ''' Constructor '''
        # CIA
        self.check_params(params)
        self.init_params()
        self.assign_params(params)

    def check_params(self, params):
        ''' Check parameters '''
        # Check params of mother class
        self.check(params)

    def init_params(self):
        ''' Initialize parameters '''
        # Init params of mother class
        self.init()

        # Init param of daughter class
        self.breath_gain_               = 243
        self.threshold_sat_             = 17000
        self.min_clean_window_          = 20
        self.threshold_std_             = 5
        self.threshold_peak_            = -1e4
        self.sec_                       = 0.1
        self.amp_min_                   = 40  # 100
        
        self.indicators_2_              = []
        self.sig_clean_2_               = []
        self.times_clean_2_             = []
        
        self.peaks_                     = []
        self.peaks_times_               = []
        self.peaks_amps_                = []
        self.peaks_amps_mv_             = []
        
        self.valleys_                   = []
        self.valleys_times_             = []
        self.valleys_amps_              = []
        self.valleys_amps_mv_           = []
        
        self.rr_                        = []
        self.rpm_                       = []
        self.rpm_var_                   = []
        self.rpm_s_                     = []
        self.rpm_var_s_                 = []
        self.times_indicators_seconds_  = []
        self.indicators_seconds_        = []
        
        self.rate_pm_times_start_       = []
        self.rate_pm_times_stop_        = []


    def assign_params(self, params):
        ''' Assign parameters '''
        # Assign params of mother class
        self.assign(params)

    def filt(self):
        """ Filter signal """
        sig = self.sig_
        fs = self.fs_
       
        sig_filt = filter_breath_unwrap(sig, fs)
       
        self.sig_filt_      = set_list_of_list(sig_filt)
        self.times_filt_    = self.times_
       
        self.flag_filt_ = True

    #### ---- The new clean ---- 
    def clean(self):
        """ Remove noise from breath signal """

        if len(self.sig_) == 0:
            print('---------------------------------------------------------')
            print('Clean: ' + self.alias_ + ' signal is empty')
            return

        sig     = self.sig_filt_
        times   = self.times_
        fs      = self.fs_    
                
        # ----- Peaks Detection with methode pylife, not used anymore -----
        # print('Peaks detection with pylife')
        # mpd = int(fs*1)
        # py_peaks, py_amps = detect_peaks_unwrap(sig, mph=0, mpd=mpd, threshold=0, edge='rising',
        #              kpsh=False, valley=False)
        # mpd     = int(fs*1)
        # py_peaks, py_amps = detect_valid_peaks_breath_unwrap(sig, fs, py_peaks)

        # ----- The old Clean method ----
        # times_clean, sig_clean,\
        #     indicators_clean = remove_noise_peak_valley_unwrap(times,
        #                                                        sig,
        #                                                        fs, 
        #                                                        peaks, 
        #                                                        peaks_amps,
        #                                                        window_time=30, 
        #                                                        n_peak_min=4,
        #                                                        period_max=8,
        #                                                        strict=True)
        
        # ---- Clean 1: Clean for breath rate processing ----
        times_clean, sig_clean, indicators_clean = remove_no_rsp_signal_unwrap(
                                                    times,
                                                    sig,
                                                    fs, 
                                                    window_s=8,
                                                    rsp_amp_min = 3 # changed from 2 to 3 as it did let too much noise in not worn tshirt
                                                    )
        
        # ---- Peaks detection inspired from neurokit (Khodadad) ----
        peaks, valleys,\
            peaks_amps, valleys_amps = detect_rsp_peaks_unwrap(sig_clean, fs)

        # ---- Clean 2: Clean for Apnea detection ----
        peaks_bis, _, peaks_amps_bis, _= detect_rsp_peaks_unwrap(sig, fs)
        _, _,\
            indicators_clean_bis = remove_noise_peak_valley_unwrap(times,
                                                                   sig,
                                                                   fs, 
                                                                   peaks_bis, 
                                                                   peaks_amps_bis,
                                                                   window_time=30, 
                                                                   n_peak_min=4,
                                                                   period_max=8,
                                                                   strict=False)
        
        indicators_clean            = set_list_of_list(indicators_clean)
        indicators_clean_bis        = set_list_of_list(indicators_clean_bis)
       
        times_indicators_seconds, indicators_seconds\
            = transform_indicators_seconds(times,
                                           indicators_clean,
                                           fs)
        if len(indicators_seconds) > 1:
            indicators_frequency = (times_indicators_seconds[1] -\
                                    times_indicators_seconds[0]) / np.timedelta64(1, 's')
            indicators_frequency = 1/indicators_frequency
        else:
            indicators_frequency = None
      
        sig_clean           = set_list_of_list(sig_clean)
        times_clean         = set_list_of_list(times_clean)
        
        # Properties
        self.clean_step_                = 1
        
        self.sig_clean_                 = sig_clean 
        self.times_clean_               = times_clean
        
        self.indicators_clean_          = indicators_clean
        self.indicators_clean_bis_      = indicators_clean_bis
            
        self.set_stats_clean_sig()
        self.set_stats_not_clean_sig()
        self.flag_clean_                = True
        self.times_indicators_seconds_  = times_indicators_seconds
        self.indicators_seconds_        = indicators_seconds
        self.indicators_frequency_      = indicators_frequency
        
        self.peaks_                     = peaks
        self.valleys_                   = valleys
        self.peaks_times_               = get_peaks_times_unwrap(times_clean, peaks)
        self.valleys_times_             = get_peaks_times_unwrap(times_clean, valleys)
        self.peaks_amps_                = peaks_amps
        self.valleys_amps_              = valleys_amps
        self.peaks_amps_mv_             = convert_mv_unwrap(self.peaks_amps_, self.breath_gain_)    # do we need this ?
        self.valleys_amps_mv_           = convert_mv_unwrap(self.valleys_amps_, self.breath_gain_)  # do we need this ?
    
    # --- The new analyze ---
    def analyze(self, on_sig=None):
        """ Analyze Breath """
        
        if on_sig is None:
            on_sig = 'clean_' + str(self.clean_step_)
        self.flag_analyze_ = True
        times, sig, indicators_clean = self.select_on_sig(on_sig)
        if len(sig) == 0:
            print('--------------------------------------------------')
            print('Analyze: ' + self.alias_ + ' signal ' + on_sig
                  + ' is empty')
            return
 
        fs = self.fs_
        
        peaks_times       = unwrap(self.peaks_times_) 
        valleys_times     = unwrap(self.valleys_times_) 
        
        peaks_times_clean, peaks_amps_clean, \
            peaks_clean   = get_rsp_peaks_clean_unwrap(times, sig, peaks_times)
            
        valleys_times_clean, valleys_amps_clean,\
            valleys_clean = get_rsp_peaks_clean_unwrap(times, sig, valleys_times)
            
        self.peaks_clean_         = peaks_clean
        self.valleys_clean_       = valleys_clean
        self.peaks_amps_clean_    = peaks_amps_clean
        self.valleys_amps_clean_  = valleys_amps_clean
        self.peaks_times_clean_   = peaks_times_clean
        self.valleys_times_clean_ = valleys_times_clean
        
        rsp_features = compute_rsp_features_unwrap(
                        sig,
                        peaks_clean, valleys_clean, 
                        peaks_amps_clean, valleys_amps_clean, 
                        peaks_times_clean, valleys_times_clean, fs)
        
        times_rr                  = rsp_features['rsp_cycles_times']
        rr_intervals              = rsp_features['rsp_rr_intervals']
        self.times_rr_            = times_rr
        self.rr_                  = rr_intervals
        self.rsp_features_        = rsp_features
        
        
        if is_list_of_list(rr_intervals):
            rr_intervals = unwrap(rr_intervals)
        
        if len(rr_intervals) > 0:
            # RPM: respirations per minute "how many cycles in a minute?"
            self.rpm_       = [60/np.mean(np.array(rr_intervals))] 
            
            # RPM var: "variation of number of cycles in a minute"
            self.rpm_var_   = [np.std(60/(np.array(rr_intervals)))]        
        
            # Breath rate (in seconds): "how much time takes a breath cycle?"
            self.rpm_s_     = [np.mean(rr_intervals)] 
            
            # "Variation of lengh breath cycles (in seconds)"
            self.rpm_var_s_ = [np.std(rr_intervals)]
                    
        self.analyze_on_sig_ = on_sig
        
    # --- The old analyze ---
    # def analyze(self, on_sig=None):
    #     """ Analyze Breath """
        
    #     if on_sig is None:
    #         on_sig = 'clean_' + str(self.clean_step_)
    #     self.flag_analyze_ = True
    #     times, sig, indicators_clean = self.select_on_sig(on_sig)
    #     if len(sig) == 0:
    #         print('--------------------------------------------------')
    #         print('Analyze: ' + self.alias_ + ' signal ' + on_sig
    #               + ' is empty')
    #         return
 
    #     fs = self.fs_
        
    #     peaks_times     = unwrap(self.peaks_times_) 
    #     peaks_times_clean, peaks_clean = get_peaks_clean_unwrap(times, 
    #                                                     sig, 
    #                                                     peaks_times)
        
     
    #     # rr_intervals = pos_2_time_interval_unwrap(fs, peaks_clean)

    #     self.peaks_clean_         = peaks_clean
    #     self.peaks_times_clean_   = get_peaks_times_unwrap(times, peaks_clean)
        
    #     rr_features                 = compute_rr_features_unwrap(peaks_times_clean, 
    #                                                               peaks_clean, 
    #                                                               fs)
        
    #     times_rr                    = rr_features['times_rr']
    #     rr_intervals                = rr_features['rr']
    #     self.times_rr_              = times_rr
    #     self.rr_                    = rr_intervals
        
    #     if is_list_of_list(rr_intervals):
    #         rr_intervals = unwrap(rr_intervals)
        
    #     # Breathing rate info in rpm
    #     if len(rr_intervals) > 0:
    #         self.rpm_       = [60/np.mean(np.array(rr_intervals)/1e3)]
    #         self.rpm_var_   = [np.std(60/(np.array(rr_intervals)/1e3))]
        
    #         # Breathing rate info in s
    #         self.rpm_s_     = [np.mean(rr_intervals)]
    #         self.rpm_var_s_ = [np.std(rr_intervals)]
                    
    #     self.analyze_on_sig_ = on_sig
    
###############################################################################
class Breath_1(Breath):
    """ Thoracic breath class """
    name_ = 'Breath thoracic'
    alias_ = 'breath_1'


###############################################################################
class Breath_2(Breath):
    """ Abdominal breath class """
    name_ = 'Breath abdominal'
    alias_ = 'breath_2'


###############################################################################
class Breaths():
    """ Breaths class """
    name_ = 'Breaths'
    alias_ = 'breaths'

    def __init__(self, breath_1, breath_2):
        """ Constructor

        Parameters
        ---------------
        breath_1, Breath_1 object
        breath_2: Breath_2 object

        """
        self.check_params(breath_1, breath_2)
        self.init_params()
        self.assign_params(breath_1, breath_2)

    def check_params(self, breath_1, breath_2):

        self.flag_error_    = False
        self.error_names_   = []
        self.log_           = []
        
        if breath_1.is_empty_:
            self.flag_error_= True
            self.error_names_.append('BREATH_1_IS_EMPTY')
        if breath_2.is_empty_:
            self.flag_error_= True
            self.error_names_.append('BERATH_2_IS_EMPTY')
            
        if self.flag_error_:
            return
        
        sig_1   = breath_1.sig_
        times_1 = breath_1.times_
        
        sig_2   = breath_2.sig_
        times_2 = breath_2.times_

        for id_seg in range(min([len(sig_1), len(sig_2)])):
            if len(sig_1[id_seg]) != len(sig_2[id_seg]):
                self.flag_error_ = True
                self.error_names_.append('BREATH_LENGTH')
                log = str('\n-------------------------------------------------' +\
                      '\n!!! WARNING !!! Error length breaths' +\
                      '\nSegment n° ' + str(id_seg) +\
                      '\nTime temp_1:' + str(times_1[id_seg][0]) +\
                      '\nLength temp_1:' + str(len(sig_1[id_seg])) +\
                      '\nTime temp_2:' + str(times_2[id_seg][0]) +\
                      '\nLength temp_2:' + str(len(sig_2[id_seg]))
                      )
                print(log)
                self.log_.append(log)
#             
    def init_params(self):
        pass

    def assign_params(self, breath_1, breath_2):
        pass

###############################################################################
class ECG(SiglifePeriodic):
    """ Electrocardiogram (ECG) class """
    name_               = 'ECG'
    alias_              = 'ecg'

    def __init__(self, params):
        ''' Constructor '''
        # CIA
        self.check_params(params)
        self.init_params()
        self.assign_params(params)

    def check_params(self, params):
        ''' Check parameters '''
        # Check params of mother class
        self.check(params)

    def init_params(self):
        ''' Initialize parameters '''
        # Init params of mother class
        self.init()
        # Init param of daughter class

        self.min_clean_window_          = 55

        self.threshold_sat_             = 100000
        self.threshold_std_             = 0
        self.threshold_peak_            = -1e4
        self.sec_                       = 0.1
        self.ecg_gain_                  = 204

        self.peaks_                     = []
        self.peaks_times_               = []
        self.peaks_qrs_                 = []
        self.peaks_qrs_times_           = []
        self.peaks_clean_               = []
        self.peaks_times_clean_         = []
        self.peaks_qrs_clean_           = []
        self.peaks_qrs_times_clean_     = []
        self.peaks_amps_                = []
        self.peaks_amps_mv_             = []
        self.peaks_qrs_                 = []
        
        self.times_rr_                  = []
        self.rr_                        = []
        
        self.sdnn_                      = []
        self.rmssd_                     = []
        self.lnrmssd_                   = []
        self.pnn50_                     = []
        
        self.bpm_                       = []
        self.bpm_var_                   = []
       
        self.bpm_ms_                    = []
        self.bpm_var_ms_                = []
        
        
       
        self.sig_mv_                    = []
        self.sig_filt_mv_               = []
        self.times_indicators_seconds_  = []
        self.indicators_seconds_        = []
        self.indicators_frequency_      = []
        self.threshold_std2_            = 20000
        self.add_condition_ = False
        
        self.rate_pm_times_start_       = []
        self.rate_pm_times_stop_        = []


        self.q_start_index_             = []
        self.q_start_time_              = []   
        self.t_stop_index_              = []  
        self.t_stop_time_               = []  
        self.qt_length_                 = []
        self.qt_length_unwrap_          = []     
        self.qt_length_mean_            = []   
        self.qt_length_median_          = [] 
        self.qt_length_median_corrected_= []
        self.qt_length_std_             = []
        self.qt_c_framingham_           = []
        self.qt_c_framingham_per_seg_   = []


        
    def assign_params(self, params):
        ''' Assign parameters '''
        # Assign params of mother class
        self.assign(params)

        self.ecg_gain_         = 204
        
        if self.device_model_ == 'brassiere':
            self.threshold_sat_     = 400000
            self.threshold_std_     = 1000
            self.threshold_peak_    = 4e3
            self.sec_ = 0.1
        else:
            if 0 < self.card_version_ <= 704:
                 self.threshold_sat_    = 100000
                 self.threshold_std_    = 0       
                 self.threshold_peak_   = 0
                 self.threshold_std2_   = 1E6
                 self.sec_              = 0.1
                 # self.ecg_gain_         = 824
                 self.min_clean_window_ = 5
                 self.add_condition_    = False
            
            elif self.card_version_ == 705:
                 self.threshold_sat_    = 15000
                 self.threshold_std_    = 0
                 self.threshold_peak_   = 0
                 self.threshold_std2_   = 5000
                 self.sec_              = 2
                 # self.ecg_gain_         = 204
                 self.min_clean_window_ = 59
                 self.add_condition_    = True
            else:
                return
            
        if not self.is_empty_:
            self.sig_mv_ = convert_mv_unwrap(self.sig_, self.ecg_gain_)

    def filt(self):
        """ Filter signal """
        sig = self.sig_
        fs = self.fs_

        sig_filt            = filter_ecg_scipy_unwrap(sig, fs)
        self.sig_filt_      = set_list_of_list(sig_filt)
        self.sig_filt_mv_   = convert_mv_unwrap(self.sig_filt_, self.ecg_gain_)
        self.times_filt_    = self.times_

        self.flag_filt_ = True
        
    def clean(self):
        """ Remove noise from signal """
        
        if len(self.sig_) == 0:
            print('---------------------------------------------------------')
            print('Clean: ' + self.alias_ + ' signal is empty')
            return

        # if not self.is_fw_version_ok_:
        #     return
        
        sig     = self.sig_filt_
        times   = self.times_
        fs      = self.fs_
        
        #beg = time.time()
        times_clean_2               = []
        sig_clean_2                 = []
        indicators_clean_2          = []
        indicators_clean_2_unwrap   = np.zeros(len(unwrap(sig)))
        indicators_clean_1_2        = []
            
        times_clean_3               = []
        sig_clean_3                 = []
        indicators_clean_3          = []
        indicators_clean_3_unwrap   = np.zeros(len(unwrap(sig)))
         
        #nd = time.time()
        #print('assign values 1  ', round(nd - beg, 3), ' s')
        
        #### new peak version
        # beg = time.time()
        # peaks_qrs  = getPeaks_unwrap(sig, fs = fs)
        # amps       = unwrap_peak_ampl (sig, peaks_qrs, fs = fs)
        # peaks      = peaks_qrs
        # nd = time.time()
        # print('new peak calc time = ', round(nd - beg, 3), ' s')
        # #### old peak version
        # beg = time.time()
        # mpd = int(fs*.35)
        # peaks_qrs   = detect_qrs_unwrap(sig, fs, mpd=mpd, rm_peaks=True)
        # peaks, amps = detect_peaks_r_unwrap(sig, fs, peaks_qrs)
        # nd = time.time()
        # print('old peak calc time = ', round(nd - beg, 3), ' s')
        
        # beg = time.time()
        # times_clean, sig_clean,\
        #     indicators_clean = remove_noise_peak_valley_unwrap(times,
        #                                                         sig,
        #                                                         fs, 
        #                                                         peaks, 
        #                                                         amps,
        #                                                         window_time=10, 
        #                                                         n_peak_min=5, # change from 6 to 5 to get from 36 to 30 BPM minimum
        #                                                         period_max=2.5, # 24 BPM min for each RR
        #                                                         strict=True)
      
        # indicators_clean = set_list_of_list(indicators_clean)
        # nd = time.time()
        # print('remove_noise_peak_valley_unwrap  calc time = ', round(nd - beg, 3), ' s')
        
        ####### temporary clean version
        #beg = time.time()
        times_clean, sig_clean,\
            indicators_clean = remove_saturation_and_big_ampls_unwrap(ecg_raw = self.sig_, 
                                                                      ecg_filt = sig,
                                                                      ecg_time = times, 
                                                                      fs = 200, 
                                                                      window_s = 10, 
                                                                      sat_high = 3750,sat_low = 50,
                                                                      amp_max = 2000, amp_min = 20)

            
      
        indicators_clean = set_list_of_list(indicators_clean)
        #nd = time.time()
        #print('remove_noise_simple  calc time = ', round(nd - beg, 3), ' s')
        
        #beg = time.time()
        times_indicators_seconds, indicators_seconds\
            = transform_indicators_seconds(times,
                                           indicators_clean,
                                           fs)
        #nd = time.time()
        #print('transform_indicators_seconds  calc time = ', round(nd - beg, 3), ' s')
        
        #beg = time.time()
        peaks_qrs  = getPeaks_unwrap(sig_clean, fs = fs)
        amps       = unwrap_peak_ampl (sig_clean, peaks_qrs, fs = fs)
        peaks      = peaks_qrs
        
        #nd = time.time()
        #print('getPeaks_unwrap calc time (peaks on clean) = ', round(nd - beg, 3), ' s')
        
   
        
        # if sum(unwrap(indicators_clean)) > 0:
        #         beg = time.time()
        #         times_clean_2, sig_clean_2, indicators_clean_1_2\
        #             = remove_noise_with_emd_unwrap(times_clean,
        #                                             sig_clean,
        #                                             fs=fs,
        #                                             window=3)
            
        #         indicators_clean_1_2 = set_list_of_list(indicators_clean_1_2)
        #         nd = time.time()
        #         print('remove_noise_with_emd_unwrap calc time = ', round(nd - beg, 3), ' s')
        #         if len(times_clean_2) > 0:
        #             beg = time.time()
        #             indicators_clean_2_unwrap = np.array(unwrap(indicators_clean))
        #             indicators_clean_2_unwrap[indicators_clean_2_unwrap > 0] = np.array(unwrap(indicators_clean_1_2))
        #             nd = time.time()
        #             print('indicators_clean_2_unwrap calc time = ', round(nd - beg, 3), ' s')
        #             beg = time.time()
        #             times_clean_3, sig_clean_3, indicators_clean_3_unwrap\
        #                 = remove_noise_smooth_unwrap(times,
        #                                               sig,
        #                                               indicators_clean_2_unwrap,
        #                                               fs)
        #             nd = time.time()
        #             print('remove_noise_smooth_unwrap calc time = ', round(nd - beg, 3), ' s')
        
        # times_indicators_seconds, indicators_seconds\
        #     = transform_indicators_seconds(times,
        #                                    indicators_clean_3_unwrap,
        #                                    fs)
            
        #beg = time.time()
        if len(indicators_seconds) > 1:
            indicators_frequency = (times_indicators_seconds[1] -\
                                    times_indicators_seconds[0]) / np.timedelta64(1, 's')
            indicators_frequency = 1/indicators_frequency
        else:
            indicators_frequency = None
                        
        #imin = 0
        # beg = time.time()
        # for i in range(len(indicators_clean)):
        #     imax = imin + len(indicators_clean[i])
        #     indicators_clean_2.append(indicators_clean_2_unwrap[imin:imax])
        #     indicators_clean_3.append(indicators_clean_3_unwrap[imin:imax])
        #     imin = imax
        # nd = time.time()
        # print('indicators_clean_2.append(indicators_clean_2_unwrap[imin:imax] calc time = ', round(nd - beg, 3), ' s')
        #beg = time.time()                
        sig_clean           = set_list_of_list(sig_clean)
        times_clean         = set_list_of_list(times_clean)
        
        # sig_clean_2         = set_list_of_list(sig_clean_2)
        # times_clean_2       = set_list_of_list(times_clean_2)
        # indicators_clean_2  = set_list_of_list(indicators_clean_2)

        # sig_clean_3         = set_list_of_list(sig_clean_3)
        # times_clean_3       = set_list_of_list(times_clean_3)
        # indicators_clean_3  = set_list_of_list(indicators_clean_3)
        
        # Properties
        self.clean_step_                = 3
        self.sig_clean_                 = sig_clean
        self.times_clean_               = times_clean
        self.indicators_clean_          = indicators_clean
        
        self.sig_clean_2_               = sig_clean
        self.times_clean_2_             = times_clean
        self.indicators_clean_2_        = indicators_clean
        self.indicators_clean_1_2_      = indicators_clean
        
        self.sig_clean_3_               = sig_clean
        self.times_clean_3_             = times_clean
        self.indicators_clean_3_        = indicators_clean

            
        # self.sig_clean_2_               = sig_clean_2
        # self.times_clean_2_             = times_clean_2
        # self.indicators_clean_2_        = indicators_clean_2
        # self.indicators_clean_1_2_      = indicators_clean_1_2
        
        # self.sig_clean_3_               = sig_clean_3
        # self.times_clean_3_             = times_clean_3
        # self.indicators_clean_3_        = indicators_clean_3

        self.set_stats_clean_sig()
        self.set_stats_not_clean_sig()
        self.flag_clean_                = True
        self.times_indicators_seconds_  = times_indicators_seconds
        self.indicators_seconds_        = indicators_seconds
        self.indicators_frequency_      = indicators_frequency
        
        self.peaks_qrs_                 = peaks_qrs
        self.peaks_qrs_times_           = get_peaks_times_unwrap(times_clean, peaks_qrs)
        self.peaks_                     = peaks
        self.peaks_times_               = get_peaks_times_unwrap(times_clean, peaks)
        self.peaks_amps_                = amps
        self.peaks_amps_mv_             = convert_mv_unwrap(self.peaks_amps_, self.ecg_gain_)
        #nd = time.time()
        #print('rest.append(indicators_clean_2_unwrap[imin:imax] calc time = ', round(nd - beg, 3), ' s')
        
    def analyze(self, on_sig=None):
        """ Analyze ECG """
        
        # if not self.is_fw_version_ok_:
        #     return
        
        if on_sig is None:
            on_sig = 'clean_' + str(self.clean_step_)
        on_sig = 'clean_3'
        self.flag_analyze_ = True
        times, sig, indicators_clean = self.select_on_sig(on_sig)
        
        if len(sig) == 0:
            print('--------------------------------------------------')
            print('Analyze: ' + self.alias_ + ' signal ' + on_sig
                  + ' is empty')
            return

        fs              = self.fs_ 
        peaks_qrs_times = self.peaks_qrs_times_
        peaks_times     = unwrap(self.peaks_times_) 
        peaks_qrs_times_clean, peaks_qrs_clean = get_peaks_clean_unwrap(times, 
                                                        sig, 
                                                        peaks_qrs_times)
        peaks_times_clean, peaks_clean = get_peaks_clean_unwrap(times, 
                                                        sig, 
                                                        peaks_times)

        rr_features          = compute_rr_features_unwrap(peaks_qrs_times_clean, 
                                                          peaks_qrs_clean, 
                                                          fs)
        
        
        self.peaks_qrs_clean_         = peaks_qrs_clean
        self.peaks_qrs_times_clean_   = get_peaks_times_unwrap(times, peaks_qrs_clean)
        self.peaks_clean_             = peaks_clean
        self.peaks_times_clean_       = get_peaks_times_unwrap(times, peaks_clean)
        
        self.times_rr_          = rr_features['times_rr']
        self.rr_                = rr_features['rr']
        
     
        self.sdnn_              = rr_features['sdnn']
        self.rmssd_             = rr_features['rmssd']
        self.lnrmssd_           = rr_features['lnrmssd']
        self.pnn50_             = rr_features['pnn50']
        
        if is_list_of_list(self.rr_):
                self.rr_ = unwrap(self.rr_)
                   
        if len(self.rr_) > 0:
            self.bpm_               = [60/np.mean(self.rr_)*1000]
            self.bpm_var_           = [np.std(60/np.array(self.rr_)*1000)]
       
            self.bpm_ms_             = [np.mean(self.rr_)]
            self.bpm_var_ms_         = [np.std(self.rr_)]
        
        self.analyze_on_sig_ = on_sig
        
        
        """ get QT length """
        
        
        
        qt_features          = compute_qt_times(ecg_time_clean_list  = times, 
                                            ecg_sig_clean_list = sig, 
                                            fs = fs)
        
        if len(unwrap(qt_features['q_start_index_s'])) > 0:
            self.q_start_index_             = qt_features['q_start_index_s']   
            self.q_start_time_              = qt_features['q_start_times_s']   
            
            self.t_stop_index_              = qt_features['t_stop_index_s']   
            self.t_stop_time_               = qt_features['t_stop_times_s']   
            
            self.qt_length_                 = qt_features['qt_length']
            self.qt_length_unwrap_          = qt_features['qt_length_unwrap']      
            # /!\ qt_median is more reliable than qt_mean as errors 
            # sometimes produce big mean variations
            self.qt_length_mean_            = [qt_features['qt_length_mean']]   
            self.qt_length_median_          = [qt_features['qt_length_median']] 
            self.qt_length_median_corrected_ = [qt_features['qt_length_median_corrected']]
            self.qt_length_std_             = [qt_features['qt_length_std']]
            
            self.qt_c_framingham_           = [qt_features['qt_c_framingham']] 
            self.qt_c_framingham_per_seg_   = [qt_features['qt_c_framingham_per_seg']]
            
        

###############################################################################
class Temperature(Siglife):
    """ Temperature class """
    name_ = 'Temperature'
    alias_ = 'temp'
    
    def __init__(self, params):
        ''' Constructor '''
        # CIA
        self.check_params(params)
        self.init_params()
        self.assign_params(params)

    def check_params(self, params):
        ''' Check parameters '''
        # Check params of mother class
        self.check(params)

    def init_params(self):
        ''' Initialize parameters '''
        # Init params of mother class
        self.init()

        # Init param of daughter class
        self.sig_median_              = []
        self.sig_iqr_               = []
        self.times_median_            = []
        
        self.sig_var_median_          = []
        self.sig_var_std_           = []
        
        self.sig_clean_median_        = []
        self.sig_clean_iqr_         = []
        self.times_clean_median_      = []
        self.sig_clean_var_median_    = []
        self.sig_clean_var_iqr_     = []
        
        self.mean_                  = []
        self.sig_var_               = []
        self.sig_clean_var_         = []
        self.sig_clean_lcie_        = []
        self.times_clean_lcie_      = []
        

    def assign_params(self, params):
        ''' Assign parameters '''
        # Assign params of mother class
        self.assign(params)
        
        if self.is_empty_:
            return
        self.sig_ = self.sig_/100 # New sensors (Tshirt v2)
        sig     = self.sig_ 
        times   = self.times_
        fs      = self.fs_
        
               
        mean_ = np.mean(unwrap(sig))
        self.mean_ = mean_
        
        medians, iqrs, times_median = compute_median_iqr_window_unwrap(times, sig, fs, 
                                                        window_time=20)
        self.sig_median_    = medians 
        self.sig_iqr_       = iqrs
        self.times_median_  = times_median
        
        # Temperature variations
        variations  = []
        temp_t0     = sig[0][0]
        for seg in sig:
            variations.append(seg - temp_t0)
        self.sig_var_ = np.array(variations)
        
        # Temperature Variation Mean
        sig         = self.sig_var_
        
        mean_ = np.mean(unwrap(sig))
        self.mean_var_ = mean_
        
        medians, iqrs, _ = compute_median_iqr_window_unwrap(times, sig, fs, 
                                                        window_time=20)
        self.sig_var_median_  = medians 
        self.sig_var_std_     = iqrs
        
        
    def filt(self):
        """ Filter signal """
        self.sig_filt_      = self.sig_
        self.times_filt_    = self.times_

    def clean(self):
        """ Remove outliers from signal """
        if len(self.sig_) == 0:
            print('---------------------------------------------------------')
            print('Clean: ' + self.alias_ + ' signal is empty')
            return

        sig = self.sig_filt_
        times = self.times_
        fs = self.fs_

        times_clean, sig_clean, indicators_clean = remove_noisy_temp_unwrap(times, sig,
                                                                      fs,
                                                                      coeff_=2,
                                                                      temp_min=20,
                                                                      temp_max=40,
                                                                      lcie=False)
       
   
        sig_clean           = set_list_of_list(sig_clean)
        times_clean         = set_list_of_list(times_clean)
        indicators_clean    = set_list_of_list(indicators_clean)
        
    
        
        # Properties
        self.clean_step_            = 1
        self.sig_clean_             = sig_clean
        self.times_clean_           = times_clean
     
        self.indicators_clean_      = indicators_clean
        self.set_stats_clean_sig()
        self.set_stats_not_clean_sig()
        self.flag_clean_            = True

    def analyze(self, on_sig=None):
        """ Analyze Temperature """
        
        if on_sig is None:
            on_sig = 'clean_' + str(self.clean_step_)
            
        self.flag_analyze_ = True
        fs = self.fs_
        times, sig, _ = self.select_on_sig(on_sig)
        if len(sig) == 0:
            print('--------------------------------------------------')
            print('Analyze: ' + self.alias_ + ' signal ' + on_sig
                  + ' is empty')
            return
        
        mean_ = np.mean(unwrap(sig))
        self.mean_clean_ = mean_
        
        medians, iqrs, times_median = compute_median_iqr_window_unwrap(times, sig, fs, 
                                                        window_time=20)
        self.sig_clean_median_      = medians 
        self.sig_clean_iqr_         = iqrs
        self.times_clean_median_    = times_median
        
        # Temperature variations clean
        variations  = []
        temp_t0     = sig[0][0]
        for seg in sig:
            variations.append(seg - temp_t0)
        self.sig_clean_var_ = np.array(variations)
        
        # Temperature Variation Mean Clean 
        sig         = self.sig_clean_var_
        
        mean_ = np.mean(unwrap(sig))
        self.mean_clean_var_ = mean_
        
        medians, iqrs, _ = compute_median_iqr_window_unwrap(times, sig, fs, 
                                                        window_time=20)
        self.sig_clean_var_median_    = medians 
        self.sig_clean_var_iqr_     = iqrs
        
        

###############################################################################
class Temperature_1(Temperature):
    """ Temperature 1 class """
    name_ = 'Temperature Right'
    alias_ = 'temp_1'


###############################################################################
class Temperature_2(Temperature):
    """ Temperature 2 class """
    name_ = 'Temperature Left'
    alias_ = 'temp_2'
    
###############################################################################
class Temperatures():
    """ Temperatures class """
    name_ = 'Temperatures'
    alias_ = 'temps'

    def __init__(self, temp_1, temp_2):
        """ Constructor

        Parameters
        ---------------
        temp_1, Temperature_1 object
        temp_2: Temperature_2 object

        """
        self.check_params(temp_1, temp_2)
        self.init_params()
        self.assign_params(temp_1, temp_2)
        
    def check_params(self, temp_1, temp_2):

        self.flag_error_    = False
        self.error_names_   = []
        self.log_           = []
        
        if temp_1.is_empty_:
            self.flag_error_= True
            self.error_names_.append('TEMP_1_IS_EMPTY')
        if temp_2.is_empty_:
            self.flag_error_= True
            self.error_names_.append('TEMP_2_IS_EMPTY')
            
    def init_params(self):
        self.mean_ = 0
    
    def assign_params(self, temp_1, temp_2): 
        pass
    
###############################################################################
class Temperature_valid(Siglife):
    """ Temperature class """
    name_ = 'Temperatures valid'
    alias_ = 'temp_valid'
    
    def __init__(self, params):
        ''' Constructor '''
        # CIA
        self.check_params(params)
        self.init_params()
        self.assign_params(params)

    def check_params(self, params):
        ''' Check parameters '''
        # Check params of mother class
        self.check(params)

    def init_params(self):
        ''' Initialize parameters '''
        # Init params of mother class
        self.init()

        # Init param of daughter class
        self.mean_  = []
        self.std_   = []

    def assign_params(self, params):
        ''' Assign parameters '''
        # Assign params of mother class
        self.assign(params)

    def filt(self):
        """ Filter signal """
        self.sig_filt_      = self.sig_
        self.times_filt_    = self.times_

    def clean(self):
        """ Remove outliers from signal """
        if len(self.sig_) == 0:
            print('---------------------------------------------------------')
            print('Clean: ' + self.alias_ + ' signal is empty')
            return

        indicators_clean = [np.ones(len(self.sig_filt_))]
        
        # Properties
        self.clean_step_        = 1
        self.sig_clean_         = self.sig_filt_
        self.times_clean_       = self.times_
        self.indicators_clean_  = indicators_clean
        self.set_stats_clean_sig()
        # self.set_stats_not_clean_sig()
        self.flag_clean_        = True

    def analyze(self, on_sig=None):
        """ Analyze Temperature """

        if on_sig is None:
            on_sig = 'clean_' + str(self.clean_step_)
            
        self.flag_analyze_ = True
        times, sig, indicators_clean = self.select_on_sig(on_sig)
        if len(sig) == 0:
            print('--------------------------------------------------')
            print('Analyze: ' + self.alias_ + ' signal ' + on_sig
                  + ' is empty')
            return
        mean_ = np.mean(unwrap(sig))
        std_ = np.std(unwrap(sig))
        self.mean_ = mean_
        self.std_ = std_

###############################################################################
class Temperature_1_valid(Temperature_valid):
    """ Temperature 1 valid class """
    name_ = 'Temperature Right valid'
    alias_ = 'temp_1_valid'


###############################################################################
class Temperature_2_valid(Temperature_valid):
    """ Temperature 2 valid class """
    name_ = 'Temperature Left valid'
    alias_ = 'temp_2_valid'
    
###############################################################################
class Temperatures_valid():
    """ Temperatures valid class """
    name_ = 'Temperatures valid'
    alias_ = 'temps_valid'

    def __init__(self, temp_1_valid, temp_2_valid):
        """ Constructor

        Parameters
        ---------------
        temp_1, Temperature_1 object
        temp_2: Temperature_2 object

        """
        self.check_params(temp_1_valid, temp_2_valid)
        self.init_params()
        self.assign_params(temp_1_valid, temp_2_valid)
        
    def check_params(self, temp_1_valid, temp_2_valid):

        self.flag_error_ = False
        self.error_names_ = []
        
    def init_params(self):
        pass
    
    def assign_params(self, temp_1_valid, temp_2_valid): 
        pass
    
###############################################################################
class Impendance(Siglife):
    """ Pulmonary impedance class """
    name_ = 'Pulmonary impedance'
    alias_ = 'imp'

    def __init__(self, params):
        ''' Constructor '''
        # CIA
        self.check_params(params)
        self.init_params()
        self.assign_params(params)

    def check_params(self, params):
        ''' Check parameters '''
        # Check params of mother class
        self.check(params)

    def init_params(self):
        ''' Initialize parameters '''
        # Init params of mother class
        self.init()

    def assign_params(self, params):
        ''' Assign parameters '''
        # Assign params of mother class
        self.assign(params)

    def filt(self):
        """ Filter signal """
        self.sig_filt_ = self.sig_

    def clean(self):
        """ Remove outliers from signal """
        # Properties
        self.clean_step_ = 1
        self.sig_clean_ = self.sig_
        self.times_clean_ = self.times_
        self.indicators_clean_ = set_list_of_list(np.ones(len(self.sig_)))
        self.set_stats_clean_sig()
        self.set_stats_not_clean_sig()
        self.flag_clean_ = True

    def analyze(self, on_sig=None):
        """ Analyze Impedance """
        if on_sig is None:
            on_sig = 'clean_' + str(self.clean_step_)
        self.flag_analyze_ = True

###############################################################################
class Impedance_1(Impendance):
    """ Impedance 1 class """
    name_ = 'Impedance Right-Back Right-Front'
    alias_ = 'imp_1'

###############################################################################
class Impedance_2(Impendance):
    """ Impedance 2 class """
    name_ = 'Impedance Right-Back Left-Front'
    alias_ = 'imp_2'

###############################################################################
class Impedance_3(Impendance):
    """ Impedance 3 class """
    name_ = 'Impedance Left-Back Right-Front'
    alias_ = 'imp_3'

###############################################################################
class Impedance_4(Impendance):
    """ Impedance 4 class """
    name_ = 'Impedance Left-Back Left-Front'
    alias_ = 'imp_4'

###############################################################################
class Impedances():
    """ Impedance class """
    name_ = '4 Impedances'
    alias_ = 'imps'
    
    def __init__(self, imp_1, imp_2, imp_3, imp_4):
        """ Constructor

        Parameters
        ---------------
        temp_1, Temperature_1 object
        temp_2: Temperature_2 object

        """
        self.check_params(imp_1, imp_2, imp_3, imp_4)
        self.init_params()
        self.assign_params(imp_1, imp_2, imp_3, imp_4)
        
    def check_params(self, imp_1, imp_2, imp_3, imp_4):

        self.flag_error_ = False
        self.error_names_ = []
        
    def init_params(self):
        pass
    
    def assign_params(self, imp_1, imp_2, imp_3, imp_4): 
        self.imp_1 = imp_1
        self.imp_2 = imp_2
        self.imp_3 = imp_3
        self.imp_4 = imp_4
    
    def show(self, id_seg=None, on_sig='raw', on_indicator=None, 
             show_clean=False, from_time=None, to_time=None, color=None, 
             center=False, show_worn=False):
        self.imp_1.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                        show_clean=show_clean)
        self.imp_2.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                        show_clean=show_clean)
        self.imp_3.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                        show_clean=show_clean)
        self.imp_4.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                        show_clean=show_clean)
        leg1 = mpatches.Patch(color='C0', label=self.imp_1.name_)
        leg2 = mpatches.Patch(color='C1', label=self.imp_2.name_)
        leg3 = mpatches.Patch(color='C2', label=self.imp_3.name_)
        leg4 = mpatches.Patch(color='C3', label=self.imp_4.name_)
        plt.legend(handles=[leg1, leg2, leg3, leg4], fontsize=11)
        plt.title('Impedance', fontsize=18)
        
###############################################################################
class Battery(Siglife):
    """ Temperature class """
    def __init__(self, params):
        ''' Constructor '''
        # CIA
        self.check_params(params)
        self.init_params()
        self.assign_params(params)

    def check_params(self, params):
        ''' Check parameters '''
        # Check params of mother class
        self.check(params)

    def init_params(self):
        ''' Initialize parameters '''
        # Init params of mother class
        self.init()

    def assign_params(self, params):
        ''' Assign parameters '''
        # Assign params of mother class
        self.assign(params)

    def filt(self):
        """ Filter signal """
        self.sig_filt_ = self.sig_

    def clean(self):
        pass

    def analyze(self, on_sig='clean'):
        pass
