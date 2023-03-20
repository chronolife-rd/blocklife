from pylife.env import get_env
DEV = get_env()
import numpy as np
import pickle
# from random import randint
import os
# from datetime import datetime

from pylife.parse import check_file
from pylife.parse import json_load
from pylife.parse import get_sig_info as get_sig_info_json

# from pylife.api_functions import test_login_with_token
from pylife.api_functions import get_ids as get_ids_api
from pylife.api_functions import get_sig_info as get_sig_info_api
from pylife.api_functions import get as get_api
from pylife.api_functions import map_data as map_data_api

from pylife.api_v2_functions import get as get_api_v2

# from pylife.cosmos_functions import connect as connect_cosmos
# from pylife.cosmos_functions import get_ids as get_ids_cosmos
# from pylife.api_functions import get_sig_info as get_sig_info_cosmos
# from pylife.api_functions import get as get_cosmos
from pylife.api_functions import time_shift as time_shift_api

from pylife.simul_functions import get_sig_info as get_sig_info_simul 

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
from pylife.siglife import Temperature_1_valid
from pylife.siglife import Temperature_2_valid
from pylife.siglife import Temperatures_valid
from pylife.siglife import Impedance_1
from pylife.siglife import Impedance_2
from pylife.siglife import Impedance_3
from pylife.siglife import Impedance_4
from pylife.siglife import Impedances

from pylife.useful import unwrap
from pylife.useful import is_list_of_list
# from pylife.useful import get_stats_clean_sig_intersection
# from pylife.useful import get_stats_not_clean_sig_intersection
from pylife.useful import get_median_length_clean
# from pylife.useful import compute_signal_mean
# from pylife.useful import set_list_of_list
# from pylife.useful import transform_indicators_seconds

from pylife.time_functions import get_utc_offset
# from pylife.time_functions import get_np_datetime_info
# from pylife.time_functions import datetime_np2str
# from pylife.time_functions import datetime_str2np

# from pylife.filters import filter_LMS

from pylife.activity_measurement import compute_activity_level_unwrap
from pylife.remove import remove_timestamps_unwrap
from pylife.remove import remove_timestamps
from pylife.remove import remove_disconnection, remove_disconnection_loss
# from pylife.remove import remove_noise_smooth_unwrap
# from pylife.remove import remove_noise_still
# from pylife.detection import detect_worn

if DEV:
    # from pylife.useful import send_email
    # from pylife.useful import pdf_report
    # from pylife.show_functions import savefig_from_report
    # from pylife.show_functions import savefig_from_report_time_split
    import pandas as pd
    # from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    # import pytz


class Datalife():
    """ Get, Parse, filt, clean, analyze,
    and show signals from a given Chronolife data source """
    SIG_DURATION_PROD = 60 # seconds

    def __init__(self, params):
        pass

    def init(self):
        """ Init parameters in mother class """
        self.flag_acc_                      = False
        self.flag_breath_                   = False
        self.flag_ecg_                      = False
        self.flag_temp_                     = False
        self.flag_temp_valid_               = False
        self.flag_imp_                      = False
        self.flag_clean_                    = False
        self.flag_analyze_                  = False
        self.flag_lms                       = False
        self.flag_remove_still_             = False
        self.xls_report_name_               = 'Report.xlsx'
        self.xls_report_time_split_name_    = 'ReportTimeSplit.xlsx'
        self.xls_report_                    = None
        self.xls_report_time_split_         = None
        self.mobile_names_                  = None
        self.diagwear_names_                = None
        self.fw_versions_                    = None
        self.app_versions_                   = None
        self.time_zone_                     = None
        self.utc_offset_                    = None
        self.activity_types_                = None
        self.lms_                           = None
        self.still_times_                   = None
        self.move_times_                    = None
        self.is_worn_                       = True            
        self.errors_                        = []
        
        # Classes
        self.accx           = None
        self.accy           = None
        self.accz           = None
        self.accs           = None
        self.breath_1       = None
        self.breath_2       = None
        self.breaths        = None
        self.ecg            = None
        self.temp_1         = None
        self.temp_2         = None
        self.temps          = None
        self.temp_1_valid   = None
        self.temp_2_valid   = None
        self.temps_valid    = None
        self.imp_1          = None
        self.imp_2          = None
        self.imp_3          = None
        self.imp_4          = None
        self.imps           = None

    def check(self, params):
        pass

    def assign(self, params):
        """ Assign parameters in mother class """

        if 'flag_acc' in params.keys():
            self.flag_acc_ = params['flag_acc']

        if 'flag_breath' in params.keys():
            self.flag_breath_ = params['flag_breath']

        if 'flag_ecg' in params.keys():
            self.flag_ecg_ = params['flag_ecg']

        if 'flag_temp' in params.keys():
            self.flag_temp_ = params['flag_temp']
            
        if 'flag_temp_valid' in params.keys():
            self.flag_temp_valid_ = params['flag_temp_valid']

        if 'flag_imp' in params.keys():
            self.flag_imp_ = params['flag_imp']

        if 'device_model' in params.keys():
            self.device_model_ = params['device_model']
        else:
            self.device_model_ = 't-shirt'
            
        if 'diagwear_name' in params.keys():
            self.diagwear_name_ = params['diagwear_name']
        else:
            self.diagwear_name_ = None
            
        if 'time_zone' in params.keys():
            self.time_zone_ = params['time_zone']
            
        if 'ecg_gain' in params.keys():
            self.ecg_gain_ = params['ecg_gain']
        else:
            self.ecg_gain_ = 824
            
        if 'breath_gain' in params.keys():
            self.breath_gain_ = params['breath_gain']
        else:
            self.breath_gain_ = 243
            
        if 'verbose' in params.keys():
            self.verbose_ = params['verbose']
        else:
            self.verbose_ = 0

    def get_sig(self, signal_type):
        """ Get values and info for a given signal type """
        pass

    def parse(self, from_time=None, to_time=None):
        self.parse_sig(from_time=from_time, to_time=to_time)
        if self.verbose_ > 0:
            self.print_load_errors()

    def parse_sig(self, from_time=None, to_time=None):
        """ Load signal information """

        if from_time is None and to_time is None:
            if self.flag_acc_:
                self.accx = Acceleration_x(self.get_sig('accx'))
                self.accy = Acceleration_y(self.get_sig('accy'))
                self.accz = Acceleration_z(self.get_sig('accz'))
                if not self.accx.is_empty_ and not self.accy.is_empty_ and not self.accz.is_empty_:
                    self.accs = Accelerations(self.accx, self.accy, self.accz)

            if self.flag_breath_:
                self.breath_1 = Breath_1(self.get_sig('breath_1'))
                self.breath_2 = Breath_2(self.get_sig('breath_2'))
                if not self.breath_1.is_empty_ and not self.breath_2.is_empty_:
                    self.breaths = Breaths(self.breath_1, self.breath_2)

            if self.flag_ecg_:
                self.ecg = ECG(self.get_sig('ecg'))

            if self.flag_temp_:
                self.temp_1 = Temperature_1(self.get_sig('temp_1'))
                self.temp_2 = Temperature_2(self.get_sig('temp_2'))
                if not self.temp_1.is_empty_ and not self.temp_2.is_empty_:
                    self.temps = Temperatures(self.temp_1, self.temp_2)
                    
            if self.flag_temp_valid_:
                self.temp_1_valid = Temperature_1_valid(self.get_sig('temp_1_valid'))
                self.temp_2_valid = Temperature_2_valid(self.get_sig('temp_2_valid'))
                if not self.temp_1_valid.is_empty_ and not self.temp_2_valid.is_empty_:
                    self.temps_valid = Temperatures_valid(self.temp_1_valid, self.temp_2_valid)

            if self.flag_imp_:
                self.imp_1 = Impedance_1(self.get_sig('imp_1'))
                self.imp_2 = Impedance_2(self.get_sig('imp_2'))
                self.imp_3 = Impedance_3(self.get_sig('imp_3'))
                self.imp_4 = Impedance_4(self.get_sig('imp_4'))
                
                if not self.imp_1.is_empty_:
                    self.imps = Impedances(self.imp_1, self.imp_2, self.imp_3, self.imp_4)

        else:
            if self.flag_acc_:
                self.accx_fromto = Acceleration_x(self.get_sig_fromto(self.accx,
                                                                      from_time,
                                                                      to_time))
                self.accy_fromto = Acceleration_y(self.get_sig_fromto(self.accy,
                                                                      from_time,
                                                                      to_time))
                self.accz_fromto = Acceleration_z(self.get_sig_fromto(self.accz,
                                                                      from_time,
                                                                      to_time))
                self.accs_fromto = Accelerations(self.accx_fromto,
                                                self.accy_fromto,
                                                self.accz_fromto)

            if self.flag_breath_:
                self.breath_1_fromto = Breath_1(self.get_sig_fromto(self.breath_1,
                                                                    from_time,
                                                                    to_time))
                self.breath_2_fromto = Breath_2(self.get_sig_fromto(self.breath_2,
                                                                    from_time,
                                                                    to_time))
                self.breaths_fromto = Breaths(self.breath_1_fromto,
                                             self.breath_2_fromto)

            if self.flag_ecg_:
                self.ecg_fromto = ECG(self.get_sig_fromto(self.ecg,
                                                          from_time,
                                                          to_time))

            if self.flag_temp_:
                self.temp_1_fromto  = Temperature_1(self.get_sig_fromto(self.temp_2, from_time, to_time))
                self.temp_2_fromto  = Temperature_2(self.get_sig_fromto(self.temp_1, from_time, to_time))
                self.temps_fromto   = Temperatures(self.temp_1_fromto,
                                                 self.temp_2_fromto)
                
            if self.flag_temp_valid_:
                self.temp_1_valid_fromto  = Temperature_1_valid(self.get_sig_fromto(self.temp_2_valid, from_time, to_time))
                self.temp_2_valid_fromto  = Temperature_2_valid(self.get_sig_fromto(self.temp_1_valid, from_time, to_time))
                self.temps_valid_fromto   = Temperatures_valid(self.temp_1_valid_fromto,
                                                 self.temp_2_valid_fromto)

            if self.flag_imp_:
                self.imp_1_fromto = Impedance_1(self.get_sig_fromto(self.imp_1,
                                                                    from_time,
                                                                    to_time))
                self.imp_2_fromto = Impedance_2(self.get_sig_fromto(self.imp_2,
                                                                    from_time,
                                                                    to_time))
                self.imp_3_fromto = Impedance_3(self.get_sig_fromto(self.imp_3,
                                                                    from_time,
                                                                    to_time))
                self.imp_4_fromto = Impedance_4(self.get_sig_fromto(self.imp_4,
                                                                    from_time,
                                                                    to_time))
                self.imps_fromto  = Impedances(self.imp_1_fromto,
                                               self.imp_2_fromto,
                                               self.imp_3_fromto,
                                               self.imp_4_fromto)

    def print_load_errors(self):
        print('--------------------------------------------------------------')
        print('ERRORS FROM DATA LOADED')
        count_errors = 0
        if self.flag_acc_:
            if self.accs is not None:
                if self.accs.flag_error_:
                    print('acc', self.accs.error_names_)
                    count_errors += 1
            if self.accx.flag_error_:
                print('accx', self.accx.error_names_)
                count_errors += 1
            if self.accy.flag_error_:
                print('accy', self.accy.error_names_)
                count_errors += 1
            if self.accz.flag_error_:
                print('accz', self.accz.error_names_)
                count_errors += 1
                
        if self.flag_breath_ :
            if self.breaths is not None:
                if self.breaths.flag_error_:
                    print('breaths', self.breaths.error_names_)
                    count_errors += 1
            if self.breath_1 is not None:
                if self.breath_1.flag_error_:
                    print('breath_1', self.breath_1.error_names_)
                    count_errors += 1
            if self.breath_2 is not None:
                if self.breath_2.flag_error_:
                    print('breath_2', self.breath_2.error_names_)
                    count_errors += 1
                
        if self.flag_ecg_:
            if self.ecg.flag_error_:
                print('ecg', self.ecg.error_names_)    
                count_errors += 1
        
        if self.flag_temp_ :
            if self.temps is not None:
                if self.temps.flag_error_:
                    print('temps', self.temps.error_names_)
                    count_errors += 1
            if self.temp_1 is not None:
                if self.temp_1.flag_error_:
                    print('temp_1', self.temp_1.error_names_)
                    count_errors += 1
            if self.temp_2 is not None:
                if self.temp_2.flag_error_:
                    print('temp_2', self.temp_2.error_names_)
                    count_errors += 1
                    
        if self.flag_temp_valid_ :
            if self.temps_valid is not None:
                if self.temps_valid.flag_error_:
                    print('temps_valid', self.temps_valid.error_names_)
                    count_errors += 1
            if self.temp_1_valid is not None:
                if self.temp_1_valid.flag_error_:
                    print('temp_1_valid', self.temp_1_valid.error_names_)
                    count_errors += 1
            if self.temp_2_valid is not None:
                if self.temp_2_valid.flag_error_:
                    print('temp_2_valid', self.temp_2_valid.error_names_)
                    count_errors += 1
                
        if self.flag_imp_:
            if self.imp_1 is not None:
                if self.imp_1.flag_error_:
                    print('imp_1', self.imp_1.error_names_)
                    count_errors += 1
            if self.imp_2 is not None:
                if self.imp_2.flag_error_:
                    print('imp_2', self.imp_2.error_names_)
                    count_errors += 1
            if self.imp_3 is not None:
                if self.imp_3.flag_error_:
                    print('imp_3', self.imp_3.error_names_)
                    count_errors += 1
            if self.imp_4 is not None:
                if self.imp_4.flag_error_:
                    print('imp_4', self.imp_4.error_names_)
                    count_errors += 1
        if count_errors == 0:
            print('None')
        else:
            print('Number of errors:', count_errors)
    
    def filt(self, from_time=None, to_time=None):
        """ filt signal """
        
        if from_time is None and to_time is None:
            if self.flag_acc_ and self.accs:
                if not self.accs.is_empty_:
                    self.accs.filt()

            if self.flag_breath_ and self.breath_1 is not None:
                if not self.breath_1.is_empty_:
                    self.breath_1.filt()
            if self.flag_breath_ and self.breath_2 is not None: 
                if not self.breath_2.is_empty_:
                    self.breath_2.filt()

            if self.flag_ecg_ and self.ecg:
                if not self.ecg.is_empty_:
                    self.ecg.filt()

            if self.flag_temp_ and self.temp_1 is not None:
                if not self.temp_1.is_empty_:
                    self.temp_1.filt()
            if self.flag_temp_ and self.temp_2 is not None:
                if not self.temp_2.is_empty_:
                    self.temp_2.filt()
                    
            if self.flag_temp_valid_ and self.temp_1_valid is not None:
                if not self.temp_1_valid.is_empty_:
                    self.temp_1_valid.filt()
            if self.flag_temp_valid_ and self.temp_2_valid is not None:
                if not self.temp_2_valid.is_empty_:
                    self.temp_2_valid.filt()

            if self.flag_imp_ and self.imp_1 is not None:
                if not self.imp_1.is_empty_:
                    self.imp_1.filt()
            if self.flag_imp_ and self.imp_2 is not None:
                if not self.imp_2.is_empty_:
                    self.imp_2.filt()
            if self.flag_imp_ and self.imp_3 is not None:
                if not self.imp_3.is_empty_:
                    self.imp_3.filt()
            if self.flag_imp_ and self.imp_4 is not None:
                if not self.imp_4.is_empty_:
                    self.imp_4.filt()

        else:
            if self.flag_acc_ and self.accs:
                if not self.accs_fromto.is_empty_:
                    self.accs_fromto.filt()

            if self.flag_breath_ and self.breath_1:
                if not self.breath_1_fromto.is_empty_:
                    self.breath_1_fromto.filt()
                if not self.breath_2_fromto.is_empty_:
                    self.breath_2_fromto.filt()

            if self.flag_ecg_ and self.ecg:
                if not self.ecg_fromto.is_empty_:
                    self.ecg_fromto.filt()

            if self.flag_temp_ and self.temp_1:
                if not self.temp_1_fromto.is_empty_:
                    self.temp_1_fromto.filt()
                if not self.temp_2.is_empty_:
                    self.temp_2_fromto.filt()
                    
            if self.flag_temp_valid_ and self.temp_1_valid:
                if not self.temp_1_valid_fromto.is_empty_:
                    self.temp_1_valid_fromto.filt()
                if not self.temp_2_valid.is_empty_:
                    self.temp_2_valid_fromto.filt()

            if self.flag_imp_ and self.imp_1:
                if not self.imp_1_fromto.is_empty_:
                    self.imp_1_fromto.filt()
                if not self.imp_2.is_empty_:
                    self.imp_2_fromto.filt()
                if not self.imp_3.is_empty_:
                    self.imp_3_fromto.filt()
                if not self.imp_4.is_empty_:
                    self.imp_4_fromto.filt()
                    
    def clean(self, from_time=None, to_time=None):
        """ Clean signal """
        
        if from_time is None and to_time is None:
            if self.flag_acc_ and self.accs:
                if not self.accs.is_empty_:
                    self.accs.clean()

            if self.flag_breath_ and self.breath_1 is not None:
                if not self.breath_1.is_empty_:
                    self.breath_1.clean()
            if self.flag_breath_ and self.breath_2 is not None: 
                if not self.breath_2.is_empty_:
                    self.breath_2.clean()

            if self.flag_ecg_ and self.ecg:
                if not self.ecg.is_empty_:
                    self.ecg.clean()

            if self.flag_temp_ and self.temp_1 is not None:
                if not self.temp_1.is_empty_:
                    self.temp_1.clean()
            if self.flag_temp_ and self.temp_2 is not None:
                if not self.temp_2.is_empty_:
                    self.temp_2.clean()
                    
            if self.flag_temp_valid_ and self.temp_1_valid is not None:
                if not self.temp_1_valid.is_empty_:
                    self.temp_1_valid.clean()
            if self.flag_temp_valid_ and self.temp_2_valid is not None:
                if not self.temp_2_valid.is_empty_:
                    self.temp_2_valid.clean()

            if self.flag_imp_ and self.imp_1 is not None:
                if not self.imp_1.is_empty_:
                    self.imp_1.clean()
            if self.flag_imp_ and self.imp_2 is not None:
                if not self.imp_2.is_empty_:
                    self.imp_2.clean()
            if self.flag_imp_ and self.imp_3 is not None:
                if not self.imp_3.is_empty_:
                    self.imp_3.clean()
            if self.flag_imp_ and self.imp_4 is not None:
                if not self.imp_4.is_empty_:
                    self.imp_4.clean()

        else:
            if self.flag_acc_ and self.accs:
                if not self.accs_fromto.is_empty_:
                    self.accs_fromto.clean()

            if self.flag_breath_ and self.breath_1:
                if not self.breath_1_fromto.is_empty_:
                    self.breath_1_fromto.clean()
                if not self.breath_2_fromto.is_empty_:
                    self.breath_2_fromto.clean()

            if self.flag_ecg_ and self.ecg:
                if not self.ecg_fromto.is_empty_:
                    self.ecg_fromto.clean()

            if self.flag_temp_ and self.temp_1:
                if not self.temp_1_fromto.is_empty_:
                    self.temp_1_fromto.clean()
                if not self.temp_2.is_empty_:
                    self.temp_2_fromto.clean()
                    
            if self.flag_temp_valid_ and self.temp_1_valid:
                if not self.temp_1_valid_fromto.is_empty_:
                    self.temp_1_valid_fromto.clean()
                if not self.temp_2_valid.is_empty_:
                    self.temp_2_valid_fromto.clean()

            if self.flag_imp_ and self.imp_1:
                if not self.imp_1_fromto.is_empty_:
                    self.imp_1_fromto.clean()
                if not self.imp_2.is_empty_:
                    self.imp_2_fromto.clean()
                if not self.imp_3.is_empty_:
                    self.imp_3_fromto.clean()
                if not self.imp_4.is_empty_:
                    self.imp_4_fromto.clean()

        self.flag_clean_ = True
        
        # !!! WARNING !!! This part may be temporary
        # if self.breath_1 and self.accs and self.ecg: 
            # self.is_worn()

    def analyze(self, from_time=None, to_time=None):
        """ Analyze signal """

        if from_time is None and to_time is None:
            if self.flag_acc_ and self.accs:
                if not self.accs.is_empty_:
                    self.accs.analyze()

            if self.flag_breath_ and self.breath_1 is not None:
                if not self.breath_1.is_empty_:
                    self.breath_1.analyze()
            if self.flag_breath_ and self.breath_2 is not None:
                if not self.breath_2.is_empty_:
                    self.breath_2.analyze()

            if self.flag_ecg_ and self.ecg:
                if not self.ecg.is_empty_:
                    self.ecg.analyze()

            if self.flag_temp_ and self.temp_1:
                if not self.temp_1.is_empty_:
                    self.temp_1.analyze()
            if self.flag_temp_ and self.temp_2 is not None:
                if not self.temp_2.is_empty_:
                    self.temp_2.analyze()
                    
            if self.flag_temp_valid_ and self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    self.temp_1_valid.analyze()
            if self.flag_temp_valid_ and self.temp_2_valid is not None:
                if not self.temp_2_valid.is_empty_:
                    self.temp_2_valid.analyze()

            if self.flag_imp_ and self.imp_1:
                if not self.imp_1.is_empty_:
                    self.imp_1.analyze()
            if self.flag_imp_ and self.imp_2:
                if not self.imp_2.is_empty_:
                    self.imp_2.analyze()
            if self.flag_imp_ and self.imp_3:
                if not self.imp_3.is_empty_:
                    self.imp_3.analyze()
            if self.flag_imp_ and self.imp_4:
                if not self.imp_4.is_empty_:
                    self.imp_4.analyze()

        else:
            if self.flag_acc_ and self.accs:
                if not self.accs_fromto.is_empty_:
                    self.accs_fromto.analyze()

            if self.flag_breath_ and self.breath_1:
                if not self.breath_1_fromto.is_empty_:
                    self.breath_1_fromto.analyze()
                    self.breath_2_fromto.analyze()

            if self.flag_ecg_ and self.ecg:
                if not self.ecg_fromto.is_empty_:
                    self.ecg_fromto.analyze()

            if self.flag_temp_ and self.temp_1:
                if not self.temp_1_fromto.is_empty_:
                    self.temp_1_fromto.analyze()
                    self.temp_2_fromto.analyze()
                    
            if self.flag_temp_valid_ and self.temp_1_valid:
                if not self.temp_1_valid_fromto.is_empty_:
                    self.temp_1_valid_fromto.analyze()
                    self.temp_2_valid_fromto.analyze()

            if self.flag_imp_ and self.imp_1:
                if not self.imp_1_fromto.is_empty_:
                    self.imp_1_fromto.analyze()
                    self.imp_2_fromto.analyze()
                    self.imp_3_fromto.analyze()
                    self.imp_4_fromto.analyze()

        self.flag_analyze_ = True

    # def lms(self):
    #     """ Analyze signal """
    #     if self.flag_ecg_ and self.ecg and self.flag_acc_:

    #         if not self.ecg.is_empty_:
    #             times_ecg, ecg_filtered,\
    #                 indicators_clean = self.ecg.select_on_sig('clean_3')
    #             lms = []
    #             for id_ in range(len(ecg_filtered)):
    #                 times, accx, indicators_clean = self.accx.select_on_sig('raw')
    #                 times, accx = self.accx.select_on_times(times, accx,
    #                                                         times_ecg[id_][0],
    #                                                         times_ecg[id_][-1])
    #                 times, accy, indicators_clean = self.accy.select_on_sig('raw')
    #                 times, accy = self.accy.select_on_times(times, accy,
    #                                                         times_ecg[id_][0],
    #                                                         times_ecg[id_][-1])
    #                 times, accz, indicators_clean = self.accz.select_on_sig('raw')
    #                 times, accz = self.accz.select_on_times(times, accz,
    #                                                         times_ecg[id_][0],
    #                                                         times_ecg[id_][-1])
    #                 acc = np.vstack((unwrap(accx), unwrap(accy)))
    #                 acc = np.vstack((acc, unwrap(accz))).T
    #                 x_acc = np.linspace(0, len(acc)/50, len(acc))
    #                 x_new = np.linspace(0, len(ecg_filtered[id_])/200,
    #                                     len(ecg_filtered[id_]))

    #                 if len(x_acc) > 0:
    #                     acc = np.array(np.interp(x_new, x_acc, acc[:, 0]))
    #                     if len(acc) == len(ecg_filtered[id_]):
    #                         y, e, w = filter_LMS(ecg_filtered[id_], acc, 10e-8)
    #                         lms.append(e)
    #                     else:
    #                         lms.append(ecg_filtered[id_])
    #                 else:
    #                     lms.append(ecg_filtered[id_])

    #             self.flag_lms   = True
    #             self.lms_       = lms
    #             self.ecg.lms_   = lms

    def show(self, signal_type='all', id_seg=None, on_sig='raw', 
             on_indicator=None, show_indicators=None, show_worn=False, 
             new_fig=True):
        """ Show signals """
        
        if not DEV:
            return
        
        if not self.flag_clean_:
            show_indicators = None

        if self.accx and not self.accx.is_empty_:
            if signal_type.lower() == self.accx.alias_ or\
                    signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.accx.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                               show_indicators=show_indicators)

        if self.accy and not self.accy.is_empty_:
            if signal_type.lower() == self.accy.alias_ or\
                    signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.accy.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                               show_indicators=show_indicators)

        if self.accz and not self.accz.is_empty_:
            if signal_type.lower() == self.accz.alias_ or\
                    signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.accz.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                               show_indicators=show_indicators)

        if self.breath_1 and not self.breath_1.is_empty_:
            if signal_type.lower() == self.breath_1.alias_ or\
                   signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.breath_1.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                   show_indicators=show_indicators)

        if self.breath_2 and not self.breath_2.is_empty_:
            if signal_type.lower() == self.breath_2.alias_ or\
                   signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.breath_2.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                   show_indicators=show_indicators)

        if self.ecg and not self.ecg.is_empty_:
            if signal_type.lower() == self.ecg.alias_ or\
                   signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.ecg.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                              show_indicators=show_indicators)

        if self.temp_1 and not self.temp_1.is_empty_:
            if signal_type.lower() == self.temp_1.alias_ or\
                  signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.temp_1.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                 show_indicators=show_indicators)

        if self.temp_2 and not self.temp_2.is_empty_:
            if signal_type.lower() == self.temp_2.alias_ or\
                    signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.temp_2.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                 show_indicators=show_indicators)
                
        if self.temp_1_valid and not self.temp_1_valid.is_empty_:
            if signal_type.lower() == self.temp_1_valid.alias_ or\
                  signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.temp_1_valid.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                 show_indicators=show_indicators)

        if self.temp_2_valid and not self.temp_2_valid.is_empty_:
            if signal_type.lower() == self.temp_2_valid.alias_ or\
                    signal_type.lower() == 'all':
                if new_fig:
                    plt.figure()
                self.temp_2_valid.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                 show_indicators=show_indicators)        

        if self.imp_1 is not None and not self.imp_1.is_empty_:
            if new_fig:
                    plt.figure()
            if signal_type.lower() == self.imp_1.alias_ or\
                   signal_type.lower() == 'all':
                
                self.imp_1.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                show_indicators=show_indicators)
                self.imp_2.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                show_indicators=show_indicators)
                self.imp_3.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                show_indicators=show_indicators)
                self.imp_4.show(id_seg=id_seg, on_sig=on_sig, on_indicator=on_indicator,
                                show_indicators=show_indicators)
                leg1 = mpatches.Patch(color='C0', label=self.imp_1.name_)
                leg2 = mpatches.Patch(color='C1', label=self.imp_2.name_)
                leg3 = mpatches.Patch(color='C2', label=self.imp_3.name_)
                leg4 = mpatches.Patch(color='C3', label=self.imp_4.name_)
                plt.legend(handles=[leg1, leg2, leg3, leg4], fontsize=11)
                plt.title('Impedance', fontsize=14)
    
    # def is_worn(self):
        
    #     self.still_times_, self.move_times_ = detect_worn(self.breath_2, self.accs, self.ecg)
            
    #     if self.accx:
    #         if not self.accx.is_empty_:
    #             self.update_is_worn(self.accx)
    #     if self.accy:
    #         if not self.accy.is_empty_:
    #             self.update_is_worn(self.accy)
    #     if self.accz:
    #         if not self.accz.is_empty_:
    #             self.update_is_worn(self.accz)
    #     if self.breath_1:
    #         if not self.breath_1.is_empty_:
    #             self.update_is_worn(self.breath_1)
    #     if self.breath_2:
    #         if not self.breath_2.is_empty_:
    #             self.update_is_worn(self.breath_2)
    #     if self.ecg:
    #         if not self.ecg.is_empty_:
    #             self.update_is_worn(self.ecg)
    #     if self.temp_1:
    #         if not self.temp_1.is_empty_:
    #             self.update_is_worn(self.temp_1)
    #     if self.temp_2:
    #         if not self.temp_2.is_empty_:
    #             self.update_is_worn(self.temp_2)
            
    #     self.is_worn_  = self.breath_1.is_worn_
        
    # def update_is_worn(self, obj):
    #     obj.still_times_        = self.still_times_
    #     obj.move_times_         = self.move_times_
    #     imax = 0
    #     indicators_worn = []
    #     for it in range(len(obj.times_)):
    #         times_seg   = obj.times_[it]
    #         ind_worn    = np.ones(len(times_seg))
    #         for j in range(len(self.still_times_)):
            
    #             xmin = self.still_times_[j]
    #             k = np.where(self.move_times_ >= xmin)[0][0]
    #             xmax = self.move_times_[k]
    #             imin = np.where(times_seg >= xmin)[0]
    #             imax    = np.where(times_seg <= xmax)[0]
                
    #             if len(imin) > 0 and len(imax) > 0:
    #                 imin    = imin[0]
    #                 imax    = imax[-1]
    #                 ind_worn[imin:imax] = 0
                
    #         indicators_worn.extend(ind_worn)
                
    #     indicators_worn         = np.array(indicators_worn)
    #     if imax != 0:
    #         indicators_worn[imax:]  = indicators_worn[imax-1]
    #     is_worn = sum((indicators_worn)) > 80/100*len((unwrap(obj.times_)))

    #     _, indicators_worn, _  = remove_disconnection(unwrap(obj.times_), indicators_worn, obj.fs_)
    #     obj.indicators_worn_    = set_list_of_list(indicators_worn)
    #     obj.is_worn_            = is_worn
        
    def get_disconnections(self, time_format='m',
                           from_time=None, to_time=None, verbose=0):

        keys = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
                'temp_1', 'temp_2', 'temp_1_valid', 'temp_2_valid', 
                'imp_1', 'imp_2', 'imp_3', 'imp_4']
        keys2 = ['number', 'percentage', 'duration', 
                 'duration_min', 'duration_max', 'duration_median', 'duration_iqr',
                 'duration_mean', 'duration_std']
        output = {}
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = None

        if time_format == 's':
            coef = 1
        elif time_format == 'm':
            coef = 60
        elif time_format == 'h':
            coef = 3600
        elif time_format == 'd':
            coef = 3600*24
        else:
            raise NameError('time_format is not correct')

        if from_time is None and to_time is None:
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'accx'
                    output[key]['number']           = self.accx.disconnection_number_
                    output[key]['percentage']       = self.accx.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.accx.disconnection_duration_)
                    output[key]['duration_min']     = self.accx.disconnection_duration_min_
                    output[key]['duration_max']     = self.accx.disconnection_duration_max_
                    output[key]['duration_median']  = self.accx.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.accx.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.accx.disconnection_duration_mean_
                    output[key]['duration_std']     = self.accx.disconnection_duration_std_
            
            if self.accy:
                if not self.accy.is_empty_:
                    key = 'accy'
                    output[key]['number']           = self.accy.disconnection_number_
                    output[key]['percentage']       = self.accy.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.accy.disconnection_duration_)
                    output[key]['duration_min']     = self.accy.disconnection_duration_min_
                    output[key]['duration_max']     = self.accy.disconnection_duration_max_
                    output[key]['duration_median']  = self.accy.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.accy.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.accy.disconnection_duration_mean_
                    output[key]['duration_std']     = self.accy.disconnection_duration_std_

            if self.accz:
                if not self.accz.is_empty_:
                    key = 'accz'
                    output[key]['number']           = self.accz.disconnection_number_
                    output[key]['percentage']       = self.accz.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.accz.disconnection_duration_)
                    output[key]['duration_min']     = self.accz.disconnection_duration_min_
                    output[key]['duration_max']     = self.accz.disconnection_duration_max_
                    output[key]['duration_median']  = self.accz.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.accz.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.accz.disconnection_duration_mean_
                    output[key]['duration_std']     = self.accz.disconnection_duration_std_


            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    output[key]['number']           = self.breath_1.disconnection_number_
                    output[key]['percentage']       = self.breath_1.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.breath_1.disconnection_duration_)
                    output[key]['duration_min']     = self.breath_1.disconnection_duration_min_
                    output[key]['duration_max']     = self.breath_1.disconnection_duration_max_
                    output[key]['duration_median']  = self.breath_1.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.breath_1.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.breath_1.disconnection_duration_mean_
                    output[key]['duration_std']     = self.breath_1.disconnection_duration_std_
            
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    output[key]['number']           = self.breath_2.disconnection_number_
                    output[key]['percentage']       = self.breath_2.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.breath_2.disconnection_duration_)
                    output[key]['duration_min']     = self.breath_2.disconnection_duration_min_
                    output[key]['duration_max']     = self.breath_2.disconnection_duration_max_
                    output[key]['duration_median']  = self.breath_2.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.breath_2.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.breath_2.disconnection_duration_mean_
                    output[key]['duration_std']     = self.breath_2.disconnection_duration_std_
                    
            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    output[key]['number']           = self.ecg.disconnection_number_
                    output[key]['percentage']       = self.ecg.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.ecg.disconnection_duration_)
                    output[key]['duration_min']     = self.ecg.disconnection_duration_min_
                    output[key]['duration_max']     = self.ecg.disconnection_duration_max_
                    output[key]['duration_median']  = self.ecg.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.ecg.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.ecg.disconnection_duration_mean_
                    output[key]['duration_std']     = self.ecg.disconnection_duration_std_
                                                          
            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    output[key]['number']           = self.temp_1.disconnection_number_
                    output[key]['percentage']       = self.temp_1.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.temp_1.disconnection_duration_)
                    output[key]['duration_min']     = self.temp_1.disconnection_duration_min_
                    output[key]['duration_max']     = self.temp_1.disconnection_duration_max_
                    output[key]['duration_median']  = self.temp_1.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.temp_1.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1.disconnection_duration_mean_
                    output[key]['duration_std']     = self.temp_1.disconnection_duration_std_

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    output[key]['number']           = self.temp_2.disconnection_number_
                    output[key]['percentage']       = self.temp_2.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.temp_2.disconnection_duration_)
                    output[key]['duration_min']     = self.temp_2.disconnection_duration_min_
                    output[key]['duration_max']     = self.temp_2.disconnection_duration_max_
                    output[key]['duration_median']  = self.temp_2.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.temp_2.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2.disconnection_duration_mean_
                    output[key]['duration_std']     = self.temp_2.disconnection_duration_std_
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    output[key]['number']           = self.temp_1_valid.disconnection_number_
                    output[key]['percentage']       = self.temp_1_valid.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.temp_1_valid.disconnection_duration_)
                    output[key]['duration_min']     = self.temp_1_valid.disconnection_duration_min_
                    output[key]['duration_max']     = self.temp_1_valid.disconnection_duration_max_
                    output[key]['duration_median']  = self.temp_1_valid.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.temp_1_valid.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1_valid.disconnection_duration_mean_
                    output[key]['duration_std']     = self.temp_1_valid.disconnection_duration_std_

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    output[key]['number']           = self.temp_2_valid.disconnection_number_
                    output[key]['percentage']       = self.temp_2_valid.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.temp_2_valid.disconnection_duration_)
                    output[key]['duration_min']     = self.temp_2_valid.disconnection_duration_min_
                    output[key]['duration_max']     = self.temp_2_valid.disconnection_duration_max_
                    output[key]['duration_median']  = self.temp_2_valid.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.temp_2_valid.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2_valid.disconnection_duration_mean_
                    output[key]['duration_std']     = self.temp_2_valid.disconnection_duration_std_

            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp_1'
                    output[key]['number']           = self.imp_1.disconnection_number_
                    output[key]['percentage']       = self.imp_1.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.imp_1.disconnection_duration_)
                    output[key]['duration_min']     = self.imp_1.disconnection_duration_min_
                    output[key]['duration_max']     = self.imp_1.disconnection_duration_max_
                    output[key]['duration_median']  = self.imp_1.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.imp_1.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.imp_1.disconnection_duration_mean_
                    output[key]['duration_std']     = self.imp_1.disconnection_duration_std_
            
            if self.imp_2:
                if not self.imp_2.is_empty_:
                    key = 'imp_2'
                    output[key]['number']           = self.imp_2.disconnection_number_
                    output[key]['percentage']       = self.imp_2.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.imp_2.disconnection_duration_)
                    output[key]['duration_min']     = self.imp_2.disconnection_duration_min_
                    output[key]['duration_max']     = self.imp_2.disconnection_duration_max_
                    output[key]['duration_median']  = self.imp_2.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.imp_2.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.imp_2.disconnection_duration_mean_
                    output[key]['duration_std']     = self.imp_2.disconnection_duration_std_

            if self.imp_3:
                if not self.imp_3.is_empty_:
                    key = 'imp_3'
                    output[key]['number']           = self.imp_3.disconnection_number_
                    output[key]['percentage']       = self.imp_3.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.imp_3.disconnection_duration_)
                    output[key]['duration_min']     = self.imp_3.disconnection_duration_min_
                    output[key]['duration_max']     = self.imp_3.disconnection_duration_max_
                    output[key]['duration_median']  = self.imp_3.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.imp_3.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.imp_3.disconnection_duration_mean_
                    output[key]['duration_std']     = self.imp_3.disconnection_duration_std_

            if self.imp_4:
                if not self.imp_4.is_empty_:
                    key = 'imp_4'
                    output[key]['number']           = self.imp_4.disconnection_number_
                    output[key]['percentage']       = self.imp_4.disconnection_percentage_
                    output[key]['duration']         = np.sum(self.imp_4.disconnection_duration_)
                    output[key]['duration_min']     = self.imp_4.disconnection_duration_min_
                    output[key]['duration_max']     = self.imp_4.disconnection_duration_max_
                    output[key]['duration_median']  = self.imp_4.disconnection_duration_median_
                    output[key]['duration_iqr']     = self.imp_4.disconnection_duration_iqr_
                    output[key]['duration_mean']    = self.imp_4.disconnection_duration_mean_
                    output[key]['duration_std']     = self.imp_4.disconnection_duration_std_

        else:
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'accx'
                    durations_info = self.accx.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
            
            if self.accy:
                if not self.accy.is_empty_:
                    key = 'accy'
                    durations_info = self.accy.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
                        
            if self.accz:
                if not self.accz.is_empty_:
                    key = 'accz'
                    durations_info = self.accz.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
                        
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    durations_info = self.breath_1.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
                        
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    durations_info = self.breath_2.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']

            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    durations_info = self.ecg.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    durations_info = self.temp_1.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
            
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    durations_info = self.temp_2.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
                        
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    durations_info = self.temp_1_valid.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
            
            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    durations_info = self.temp_2_valid.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']


            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp_1'
                    durations_info = self.imp_1.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']
            
            if self.imp_2:
                if not self.imp_2.is_empty_:
                    key = 'imp_2'
                    durations_info = self.imp_2.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']

            if self.imp_3:
                if not self.imp_3.is_empty_:
                    key = 'imp_3'
                    durations_info = self.imp_3.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']

            if self.imp_4:
                if not self.imp_4.is_empty_:
                    key = 'imp_4'
                    durations_info = self.imp_4.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        output[key]['number']           = durations_info['disconnection_number']
                        output[key]['percentage']       = durations_info['disconnection_percentage']
                        output[key]['duration']         = durations_info['disconnection_duration']
                        output[key]['duration_min']     = durations_info['disconnection_duration_min']
                        output[key]['duration_max']     = durations_info['disconnection_duration_max']
                        output[key]['duration_median']  = durations_info['disconnection_duration_median']
                        output[key]['duration_iqr']     = durations_info['disconnection_duration_iqr']
                        output[key]['duration_mean']    = durations_info['disconnection_duration_mean']
                        output[key]['duration_std']     = durations_info['disconnection_duration_std']

        if verbose > 0:
            print('----------------------------------------------------------')
            print('SIGNAL DISCONNECTION')

            if self.accx:
                if not self.accx.is_empty_:
                    print('Acc x            %.0f disconnections, %.2f%%, %.2f %s' %
                          (output['accx']['number'],
                           output['accx']['percentage'],
                           (output['accx']['duration']/coef),
                           time_format))

            if self.breath_1:
                if not self.breath_1.is_empty_:
                    print('Breath 1         %.0f disconnections, %.2f%%, %.2f %s' %
                          (output['breath_1']['number'],
                           output['breath_1']['percentage'],
                           (output['breath_1']['duration']/coef),
                           time_format))

            if self.ecg:
                if not self.ecg.is_empty_:
                    print('ECG              %.0f disconnections, %.2f%%, %.2f %s' %
                          (output['ecg']['number'],
                           output['ecg']['percentage'],
                           (output['ecg']['duration']/coef),
                           time_format))

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    print('Temp 1           %.0f disconnections, %.2f%%, %.2f %s' %
                          (output['temp_1']['number'],
                           output['temp_1']['percentage'],
                           (output['temp_1']['duration']/coef),
                           time_format))
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    print('Temp 1 valid     %.0f disconnections, %.2f%%, %.2f %s' %
                          (output['temp_1_valid']['number'],
                           output['temp_1_valid']['percentage'],
                           (output['temp_1_valid']['duration']/coef),
                           time_format))
            
            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    print('Temp 2 valid     %.0f disconnections, %.2f%%, %.2f %s' %
                          (output['temp_2_valid']['number'],
                           output['temp_2_valid']['percentage'],
                           (output['temp_2_valid']['duration']/coef),
                           time_format))

            if self.imp_1:
                if not self.imp_1.is_empty_:
                        print('Imp 1            %.0f disconnections, %.2f%%, %.2f %s' %
                              (output['imp_1']['number'],
                               output['imp_1']['percentage'],
                               (output['imp_1']['duration']/coef),
                               time_format))

        return output

    def get_disconnections_details(self, from_time=None, to_time=None):
    
        keys = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
                'temp_1', 'temp_2', 'imp_1', 'imp_2', 'imp_3', 'imp_4']
        keys2 = ['duration', 'start', 'stop']
        output = {}
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = []
    
        if from_time is None and to_time is None:
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'accx'
                    for i in range(self.accx.disconnection_number_):
                          output[key]['start'].append(self.accx.disconnection_times_start_[i])
                          output[key]['stop'].append(self.accx.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.accx.disconnection_durations_[i]))
            
            if self.accy:
                if not self.accy.is_empty_:
                    key = 'accy'
                    for i in range(self.accy.disconnection_number_):
                          output[key]['start'].append(self.accy.disconnection_times_start_[i])
                          output[key]['stop'].append(self.accy.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.accy.disconnection_durations_[i]))
            
            if self.accz:
                if not self.accz.is_empty_:
                    key = 'accz'
                    for i in range(self.accz.disconnection_number_):
                        output[key]['start'].append(self.accz.disconnection_times_start_[i])
                        output[key]['stop'].append(self.accz.disconnection_times_stop_[i])
                        output[key]['duration'].append((self.accz.disconnection_durations_[i]))
                      
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    for i in range(self.breath_1.disconnection_number_):
                          output[key]['start'].append(self.breath_1.disconnection_times_start_[i])
                          output[key]['stop'].append(self.breath_1.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.breath_1.disconnection_durations_[i]))
                
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    for i in range(self.breath_2.disconnection_number_):
                          output[key]['start'].append(self.breath_2.disconnection_times_start_[i])
                          output[key]['stop'].append(self.breath_2.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.breath_2.disconnection_durations_[i]))
    
            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    for i in range(self.ecg.disconnection_number_):
                          output[key]['start'].append(self.ecg.disconnection_times_start_[i])
                          output[key]['stop'].append(self.ecg.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.ecg.disconnection_durations_[i]))
                                                          
            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    for i in range(self.temp_1.disconnection_number_):
                          output[key]['start'].append(self.temp_1.disconnection_times_start_[i])
                          output[key]['stop'].append(self.temp_1.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.temp_1.disconnection_durations_[i]))
            
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    for i in range(self.temp_2.disconnection_number_):
                          output[key]['start'].append(self.temp_2.disconnection_times_start_[i])
                          output[key]['stop'].append(self.temp_2.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.temp_2.disconnection_durations_[i]))
    
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp_1'
                    for i in range(self.imp_1.disconnection_number_):
                          output[key]['start'].append(self.imp_1.disconnection_times_start_[i])
                          output[key]['stop'].append(self.imp_1.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.imp_1.disconnection_durations_[i]))
                          
            if self.imp_2:
                if not self.imp_2.is_empty_:
                    key = 'imp_2'
                    for i in range(self.imp_2.disconnection_number_):
                          output[key]['start'].append(self.imp_2.disconnection_times_start_[i])
                          output[key]['stop'].append(self.imp_2.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.imp_2.disconnection_durations_[i]))
            
            if self.imp_3:
                if not self.imp_3.is_empty_:
                    key = 'imp_3'
                    for i in range(self.imp_3.disconnection_number_):
                          output[key]['start'].append(self.imp_3.disconnection_times_start_[i])
                          output[key]['stop'].append(self.imp_3.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.imp_3.disconnection_durations_[i]))
            
            if self.imp_4:
                if not self.imp_4.is_empty_:
                    key = 'imp_4'
                    for i in range(self.imp_4.disconnection_number_):
                          output[key]['start'].append(self.imp_4.disconnection_times_start_[i])
                          output[key]['stop'].append(self.imp_4.disconnection_times_stop_[i])
                          output[key]['duration'].append((self.imp_4.disconnection_durations_[i]))
        else:
            
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'accx'
                    durations_info = self.accx.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
            
            if self.accy:
                if not self.accy.is_empty_:
                    key = 'accy'
                    durations_info = self.accy.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                        
            if self.accz:
                if not self.accz.is_empty_:
                    key = 'accz'
                    durations_info = self.accz.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                              
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    durations_info = self.breath_1.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
            
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    durations_info = self.breath_2.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                              
            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    durations_info = self.ecg.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
    
            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    durations_info = self.temp_1.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                              
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    durations_info = self.temp_2.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                              
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp_1'
                    durations_info = self.imp_1.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                              
            if self.imp_2:
                if not self.imp_2.is_empty_:
                    key = 'imp_2'
                    durations_info = self.imp_2.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                     
            if self.imp_3:
                if not self.imp_3.is_empty_:
                    key = 'imp_3'
                    durations_info = self.imp_3.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                     
            if self.imp_4:
                if not self.imp_4.is_empty_:
                    key = 'imp_4'
                    durations_info = self.imp_4.get_durations_info_fromto(from_time, to_time)
                    if len(durations_info) > 0:
                        for i in range(durations_info['disconnection_number']):
                              output[key]['start'].append(durations_info['disconnection_times_start'][i])
                              output[key]['stop'].append(durations_info['disconnection_times_stop'][i])
                              output[key]['duration'].append((durations_info['disconnection_durations'][i]))
                              
        return output

    def get_sig_duration(self, from_time=None, to_time=None, time_format='m',
                         verbose=0):

        keys = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
                'temp_1', 'temp_2', 'temp_1_valid', 'temp_2_valid', 
                'imp_1', 'imp_2', 'imp_3', 'imp_4']
        keys2 = ['duration', 'duration_min', 'duration_max', 
                 'duration_median', 'duration_iqr', 'duration_mean', 'duration_std']
        output = {}
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = [] 

        if time_format == 's':
            coef = 1
        elif time_format == 'm':
            coef = 60
        elif time_format == 'h':
            coef = 3600
        elif time_format == 'd':
            coef = 3600*24
        else:
            raise NameError('time_format is not correct')

        if from_time is None and to_time is None:
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'accx'
                    output[key]['duration']         = self.accx.sig_duration_
                    output[key]['duration_min']     = self.accx.sig_duration_min_
                    output[key]['duration_max']     = self.accx.sig_duration_max_
                    output[key]['duration_median']  = self.accx.sig_duration_median_
                    output[key]['duration_iqr']     = self.accx.sig_duration_iqr_
                    output[key]['duration_mean']    = self.accx.sig_duration_mean_
                    output[key]['duration_std']     = self.accx.sig_duration_std_
                    
            if self.accy:
                if not self.accy.is_empty_:
                    key = 'accy'
                    output[key]['duration']         = self.accy.sig_duration_
                    output[key]['duration_min']     = self.accy.sig_duration_min_
                    output[key]['duration_max']     = self.accy.sig_duration_max_
                    output[key]['duration_median']  = self.accy.sig_duration_median_
                    output[key]['duration_iqr']     = self.accy.sig_duration_iqr_
                    output[key]['duration_mean']    = self.accy.sig_duration_mean_
                    output[key]['duration_std']     = self.accy.sig_duration_std_
                    
            if self.accz:
                if not self.accz.is_empty_:
                    key = 'accz'
                    output[key]['duration']         = self.accz.sig_duration_
                    output[key]['duration_min']     = self.accz.sig_duration_min_
                    output[key]['duration_max']     = self.accz.sig_duration_max_
                    output[key]['duration_median']  = self.accz.sig_duration_median_
                    output[key]['duration_iqr']     = self.accz.sig_duration_iqr_
                    output[key]['duration_mean']    = self.accz.sig_duration_mean_
                    output[key]['duration_std']     = self.accz.sig_duration_std_

            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    output[key]['duration']         = self.breath_1.sig_duration_
                    output[key]['duration_min']     = self.breath_1.sig_duration_min_
                    output[key]['duration_max']     = self.breath_1.sig_duration_max_
                    output[key]['duration_median']  = self.breath_1.sig_duration_median_
                    output[key]['duration_iqr']     = self.breath_1.sig_duration_iqr_
                    output[key]['duration_mean']    = self.breath_1.sig_duration_mean_
                    output[key]['duration_std']     = self.breath_1.sig_duration_std_
            
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    output[key]['duration']         = self.breath_2.sig_duration_
                    output[key]['duration_min']     = self.breath_2.sig_duration_min_
                    output[key]['duration_max']     = self.breath_2.sig_duration_max_
                    output[key]['duration_median']  = self.breath_2.sig_duration_median_
                    output[key]['duration_iqr']     = self.breath_2.sig_duration_iqr_
                    output[key]['duration_mean']    = self.breath_2.sig_duration_mean_
                    output[key]['duration_std']     = self.breath_2.sig_duration_std_
                    
            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    output[key]['duration']         = self.ecg.sig_duration_
                    output[key]['duration_min']     = self.ecg.sig_duration_min_
                    output[key]['duration_max']     = self.ecg.sig_duration_max_
                    output[key]['duration_median']  = self.ecg.sig_duration_median_
                    output[key]['duration_iqr']     = self.ecg.sig_duration_iqr_
                    output[key]['duration_mean']    = self.ecg.sig_duration_mean_
                    output[key]['duration_std']     = self.ecg.sig_duration_std_

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    output[key]['duration']         = self.temp_1.sig_duration_
                    output[key]['duration_min']     = self.temp_1.sig_duration_min_
                    output[key]['duration_max']     = self.temp_1.sig_duration_max_
                    output[key]['duration_median']  = self.temp_1.sig_duration_median_
                    output[key]['duration_iqr']     = self.temp_1.sig_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1.sig_duration_mean_
                    output[key]['duration_std']     = self.temp_1.sig_duration_std_
            
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    output[key]['duration']         = self.temp_2.sig_duration_
                    output[key]['duration_min']     = self.temp_2.sig_duration_min_
                    output[key]['duration_max']     = self.temp_2.sig_duration_max_
                    output[key]['duration_median']  = self.temp_2.sig_duration_median_
                    output[key]['duration_iqr']     = self.temp_2.sig_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2.sig_duration_mean_
                    output[key]['duration_std']     = self.temp_2.sig_duration_std_
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    output[key]['duration']         = self.temp_1_valid.sig_duration_
                    output[key]['duration_min']     = self.temp_1_valid.sig_duration_min_
                    output[key]['duration_max']     = self.temp_1_valid.sig_duration_max_
                    output[key]['duration_median']  = self.temp_1_valid.sig_duration_median_
                    output[key]['duration_iqr']     = self.temp_1_valid.sig_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1_valid.sig_duration_mean_
                    output[key]['duration_std']     = self.temp_1_valid.sig_duration_std_
            
            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    output[key]['duration']         = self.temp_2_valid.sig_duration_
                    output[key]['duration_min']     = self.temp_2_valid.sig_duration_min_
                    output[key]['duration_max']     = self.temp_2_valid.sig_duration_max_
                    output[key]['duration_median']  = self.temp_2_valid.sig_duration_median_
                    output[key]['duration_iqr']     = self.temp_2_valid.sig_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2_valid.sig_duration_mean_
                    output[key]['duration_std']     = self.temp_2_valid.sig_duration_std_
                    
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp_1'
                    output[key]['duration']         = self.imp_1.sig_duration_
                    output[key]['duration_min']     = self.imp_1.sig_duration_min_
                    output[key]['duration_max']     = self.imp_1.sig_duration_max_
                    output[key]['duration_median']  = self.imp_1.sig_duration_median_
                    output[key]['duration_iqr']     = self.imp_1.sig_duration_iqr_
                    output[key]['duration_mean']    = self.imp_1.sig_duration_mean_
                    output[key]['duration_std']     = self.imp_1.sig_duration_std_
            
            if self.imp_2:
                if not self.imp_2.is_empty_:
                    key = 'imp_2'
                    output[key]['duration']         = self.imp_2.sig_duration_
                    output[key]['duration_min']     = self.imp_2.sig_duration_min_
                    output[key]['duration_max']     = self.imp_2.sig_duration_max_
                    output[key]['duration_median']  = self.imp_2.sig_duration_median_
                    output[key]['duration_iqr']     = self.imp_2.sig_duration_iqr_
                    output[key]['duration_mean']    = self.imp_2.sig_duration_mean_
                    output[key]['duration_std']     = self.imp_2.sig_duration_std_
                    
            if self.imp_3:
                if not self.imp_3.is_empty_:
                    key = 'imp_3'
                    output[key]['duration']         = self.imp_3.sig_duration_
                    output[key]['duration_min']     = self.imp_3.sig_duration_min_
                    output[key]['duration_max']     = self.imp_3.sig_duration_max_
                    output[key]['duration_median']  = self.imp_3.sig_duration_median_
                    output[key]['duration_iqr']     = self.imp_3.sig_duration_iqr_
                    output[key]['duration_mean']    = self.imp_3.sig_duration_mean_
                    output[key]['duration_std']     = self.imp_3.sig_duration_std_
                    
            if self.imp_4:
                if not self.imp_4.is_empty_:
                    key = 'imp_4'
                    output[key]['duration']         = self.imp_4.sig_duration_
                    output[key]['duration_min']     = self.imp_4.sig_duration_min_
                    output[key]['duration_max']     = self.imp_4.sig_duration_max_
                    output[key]['duration_median']  = self.imp_4.sig_duration_median_
                    output[key]['duration_iqr']     = self.imp_4.sig_duration_iqr_
                    output[key]['duration_mean']    = self.imp_4.sig_duration_mean_
                    output[key]['duration_std']     = self.imp_4.sig_duration_std_
                    
        else:
            if self.accx:
                if not self.accx.is_empty_:
                    info = self.accx.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'accx'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
            
            if self.accy:
                if not self.accy.is_empty_:
                    info = self.accy.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'accy'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
                        
            if self.accz:
                if not self.accz.is_empty_:
                    info = self.accz.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'accz'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
                        
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    info = self.breath_1.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'breath_1'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
                        
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    info = self.breath_2.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'breath_2'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']

            if self.ecg:
                if not self.ecg.is_empty_:
                    info = self.ecg.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'ecg'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    info = self.temp_1.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_1'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    info = self.temp_2.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_2'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
                        
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    info = self.temp_1_valid.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_1_valid'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    info = self.temp_2_valid.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_2_valid'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
                        
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    info = self.imp_1.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_1'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']

            if self.imp_2:
                if not self.imp_2.is_empty_:
                    info = self.imp_2.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_2'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
                        
            if self.imp_3:
                if not self.imp_3.is_empty_:
                    info = self.imp_3.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_3'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']
            
            if self.imp_4:
                if not self.imp_4.is_empty_:
                    info = self.imp_4.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_4'
                        output[key]['duration']         = info['sig_duration']
                        output[key]['duration_min']     = info['sig_duration_min']
                        output[key]['duration_max']     = info['sig_duration_max']
                        output[key]['duration_median']  = info['sig_duration_median']
                        output[key]['duration_iqr']     = info['sig_duration_iqr']
                        output[key]['duration_mean']    = info['sig_duration_mean']
                        output[key]['duration_std']     = info['sig_duration_std']

        if verbose > 0:
            print('-------------------------------------------------------------')
            print('SIGNAL DURATION')

            if self.accx:
                if not self.accx.is_empty_:
                    print('Accx             %.2f %s (max %.2f %s)' % ((output['accx']['duration']/coef),
                                                         time_format,
                                                         (output['accx']['duration_max']/coef),
                                                         time_format))

            if self.breath_1:
                if not self.breath_1.is_empty_:
                    print('Breath 1         %.2f %s (max %.2f %s)' % ((output['breath_1']['duration']/coef),
                                                            time_format,
                                                            (output['breath_1']['duration_max']/coef),
                                                            time_format))

            if self.ecg:
                if not self.ecg.is_empty_:
                    print('ECG              %.2f %s (max %.2f %s)' % ((output['ecg']['duration']/coef),
                                                         time_format,
                                                         (output['ecg']['duration_max']/coef),
                                                         time_format))

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    print('Temp 1           %.2f %s (max %.2f %s)' % ((output['temp_1']['duration']/coef),
                                                          time_format,
                                                          (output['temp_1']['duration_max']/coef),
                                                          time_format))
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    print('Temp 1 valid     %.2f %s (max %.2f %s)' % ((output['temp_1_valid']['duration']/coef),
                                                          time_format,
                                                          (output['temp_1_valid']['duration_max']/coef),
                                                          time_format))
                    
            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    print('Temp 2 valid     %.2f %s (max %.2f %s)' % ((output['temp_2_valid']['duration']/coef),
                                                          time_format,
                                                          (output['temp_2_valid']['duration_max']/coef),
                                                          time_format))

            if self.imp_1:
                if not self.imp_1.is_empty_:
                    print('Imp 1            %.2f %s (max %.2f %s)' % ((output['imp_1']['duration']/coef),
                                                     time_format,
                                                     (output['imp_1']['duration_max']/coef),
                                                     time_format))

        return output
                
    def get_sig_start_stop(self, from_time=None, to_time=None, verbose=0):

        keys = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
                'temp_1', 'temp_2', 'temp_1_valid', 'temp_2_valid', 
                'imp_1', 'imp_2', 'imp_3', 'imp_4']
        keys2 = ['start', 'stop']
        output = {}
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = None

        if from_time is None and to_time is None:
                    
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'accx'
                    output[key]['start']  = self.accx.time_start_
                    output[key]['stop']   = self.accx.time_stop_
                    
            if self.accy:
                if not self.accy.is_empty_:
                    key = 'accy'
                    output[key]['start']  = self.accy.time_start_
                    output[key]['stop']   = self.accy.time_stop_
                    
            if self.accz:
                if not self.accz.is_empty_:
                    key = 'accz'
                    output[key]['start']  = self.accz.time_start_
                    output[key]['stop']   = self.accz.time_stop_

            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    output[key]['start']   = self.breath_1.time_start_
                    output[key]['stop']    = self.breath_1.time_stop_
                    
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    output[key]['start']   = self.breath_2.time_start_
                    output[key]['stop']    = self.breath_2.time_stop_

            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    output[key]['start']  = self.ecg.time_start_
                    output[key]['stop']   = self.ecg.time_stop_

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    output[key]['start'] = self.temp_1.time_start_
                    output[key]['stop']  = self.temp_1.time_stop_
                    
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    output[key]['start'] = self.temp_2.time_start_
                    output[key]['stop']  = self.temp_2.time_stop_
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    output[key]['start'] = self.temp_1_valid.time_start_
                    output[key]['stop']  = self.temp_1_valid.time_stop_
                    
            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    output[key]['start'] = self.temp_2_valid.time_start_
                    output[key]['stop']  = self.temp_2_valid.time_stop_

            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp_1'
                    output[key]['start']  = self.imp_1.time_start_
                    output[key]['stop']   = self.imp_1.time_stop_
                    
            if self.imp_2:
                if not self.imp_2.is_empty_:
                    key = 'imp_2'
                    output[key]['start']  = self.imp_2.time_start_
                    output[key]['stop']   = self.imp_2.time_stop_
                    
            if self.imp_3:
                if not self.imp_3.is_empty_:
                    key = 'imp_3'
                    output[key]['start']  = self.imp_3.time_start_
                    output[key]['stop']   = self.imp_3.time_stop_
                    
            if self.imp_4:
                if not self.imp_4.is_empty_:
                    key = 'imp_4'
                    output[key]['start']  = self.imp_4.time_start_
                    output[key]['stop']   = self.imp_4.time_stop_
            
        else:
            if self.accx:
                if not self.accx.is_empty_:
                    info = self.accx.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'accx'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']
                        
            if self.accy:
                if not self.accy.is_empty_:
                    info = self.accy.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'accy'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']
            
            if self.accz:
                if not self.accz.is_empty_:
                    info = self.accz.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'accz'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']
                        
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    info = self.breath_1.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'breath_1'
                        output[key]['start']   = info['time_start']
                        output[key]['stop']    = info['time_stop']
            
            if self.breath_2:
                if not self.breath_2.is_empty_:
                    info = self.breath_2.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'breath_2'
                        output[key]['start']   = info['time_start']
                        output[key]['stop']    = info['time_stop']
                        
            if self.ecg:
                if not self.ecg.is_empty_:
                    info = self.ecg.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'ecg'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    info = self.temp_1.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_1'
                        output[key]['start'] = info['time_start']
                        output[key]['stop']  = info['time_stop']
            
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    info = self.temp_2.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_2'
                        output[key]['start'] = info['time_start']
                        output[key]['stop']  = info['time_stop']
                        
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    info = self.temp_1_valid.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_1_valid'
                        output[key]['start'] = info['time_start']
                        output[key]['stop']  = info['time_stop']
            
            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    info = self.temp_2_valid.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'temp_2_valid'
                        output[key]['start'] = info['time_start']
                        output[key]['stop']  = info['time_stop']
                        
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    info = self.imp_1.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_1'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']
                        
            if self.imp_2:
                if not self.imp_2.is_empty_:
                    info = self.imp_2.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_2'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']
            
            if self.imp_3:
                if not self.imp_3.is_empty_:
                    info = self.imp_3.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_3'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']
                        
            if self.imp_4:
                if not self.imp_4.is_empty_:
                    info = self.imp_4.get_durations_info_fromto(from_time, to_time)
                    if len(info) > 0:
                        key = 'imp_4'
                        output[key]['start']  = info['time_start']
                        output[key]['stop']   = info['time_stop']

        if verbose > 0:
            print('------------------------------------------------------')
            print('SIGNAL START STOP')
            if self.accx:
                if not self.accx.is_empty_:
                    start = pd.Series(output['accx']['start']).round('s').iloc[0]
                    stop = pd.Series(output['accx']['stop']).round('s').iloc[0]
                    print('Acc x:           Start', start, 'Stop', stop)

            if self.breath_1:
                if not self.breath_1.is_empty_:
                    start = pd.Series(output['breath_1']['start']).round('s').iloc[0]
                    stop = pd.Series(output['breath_1']['stop']).round('s').iloc[0]
                    print('Breath 1:        Start', start, 'Stop', stop)

            if self.ecg:
                if not self.ecg.is_empty_:
                    start = pd.Series(output['ecg']['start']).round('s').iloc[0]
                    stop = pd.Series(output['ecg']['stop']).round('s').iloc[0]
                    print('ECG:             Start', start, 'Stop', stop)

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    start = pd.Series(output['temp_1']['start']).round('s').iloc[0]
                    stop = pd.Series(output['temp_1']['stop']).round('s').iloc[0]
                    print('Temp 1:          Start', start, 'Stop', stop)
            
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    start = pd.Series(output['temp_1_valid']['start']).round('s').iloc[0]
                    stop = pd.Series(output['temp_1_valid']['stop']).round('s').iloc[0]
                    print('Temp 1 valid:    Start', start, 'Stop', stop)
            
            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    print('Temp 2 valid:    Start', start, 'Stop', stop)
                    
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    start = pd.Series(output['imp_1']['start']).round('s').iloc[0]
                    stop = pd.Series(output['imp_1']['stop']).round('s').iloc[0]
                    print('Imp 1:           Start', start, 'Stop', stop)

        return output
    
    def get_sig_clean_stats(self, from_time=None, to_time=None, time_format='m', verbose=0):

        output = {}
        keys = ['acc', 'breath_1', 'breath_2', 'ecg', 
                'temp_1', 'temp_2', 'temp_1_valid', 'temp_2_valid',
                'imp', 'ecg_breath', 'ecg_temp', 'ecg_breath_temp']
        keys2 = ['percentage', 'duration', 'duration_min', 'duration_max',
                 'duration_median', 'duration_iqr', 'duration_mean', 'duration_std',
                 'n_segments']
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = None

        if not self.flag_clean_:
            return output

        if time_format == 's':
            coef = 1
        elif time_format == 'm':
            coef = 60
        elif time_format == 'h':
            coef = 3600
        elif time_format == 'd':
            coef = 3600*24
        else:
            raise NameError('time_format is not correct')
            
        keys = []
        if from_time is None and to_time is None:
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'acc'
                    keys.append(key)
                    output[key]['percentage']       = self.accx.sig_clean_percentage_
                    output[key]['duration']         = self.accx.sig_clean_duration_
                    output[key]['duration_min']     = self.accx.sig_clean_duration_min_
                    output[key]['duration_max']     = self.accx.sig_clean_duration_max_
                    output[key]['duration_median']  = self.accx.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.accx.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.accx.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.accx.sig_clean_duration_std_
                    output[key]['n_segments']       = self.accx.sig_clean_n_segments_
                    
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    keys.append(key)
                    output[key]['percentage']       = self.breath_1.sig_clean_percentage_
                    output[key]['duration']         = self.breath_1.sig_clean_duration_
                    output[key]['duration_min']     = self.breath_1.sig_clean_duration_min_
                    output[key]['duration_max']     = self.breath_1.sig_clean_duration_max_
                    output[key]['duration_median']  = self.breath_1.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.breath_1.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.breath_1.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.breath_1.sig_clean_duration_std_
                    output[key]['n_segments']       = self.breath_1.sig_clean_n_segments_

            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    keys.append(key)
                    output[key]['percentage']       = self.breath_2.sig_clean_percentage_
                    output[key]['duration']         = self.breath_2.sig_clean_duration_
                    output[key]['duration_min']     = self.breath_2.sig_clean_duration_min_
                    output[key]['duration_max']     = self.breath_2.sig_clean_duration_max_
                    output[key]['duration_median']  = self.breath_2.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.breath_2.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.breath_2.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.breath_2.sig_clean_duration_std_
                    output[key]['n_segments']       = self.breath_2.sig_clean_n_segments_

            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    keys.append(key)
                    output[key]['percentage']       = self.ecg.sig_clean_percentage_
                    output[key]['duration']         = self.ecg.sig_clean_duration_
                    output[key]['duration_min']     = self.ecg.sig_clean_duration_min_
                    output[key]['duration_max']     = self.ecg.sig_clean_duration_max_
                    output[key]['duration_median']  = self.ecg.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.ecg.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.ecg.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.ecg.sig_clean_duration_std_
                    output[key]['n_segments']       = self.ecg.sig_clean_n_segments_

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_1.sig_clean_percentage_
                    output[key]['duration']         = self.temp_1.sig_clean_duration_
                    output[key]['duration_min']     = self.temp_1.sig_clean_duration_min_
                    output[key]['duration_max']     = self.temp_1.sig_clean_duration_max_
                    output[key]['duration_median']  = self.temp_1.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_1.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_1.sig_clean_duration_std_
                    output[key]['n_segments']       = self.temp_1.sig_clean_n_segments_

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_2.sig_clean_percentage_
                    output[key]['duration']         = self.temp_2.sig_clean_duration_
                    output[key]['duration_min']     = self.temp_2.sig_clean_duration_min_
                    output[key]['duration_max']     = self.temp_2.sig_clean_duration_max_
                    output[key]['duration_median']  = self.temp_2.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_2.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_2.sig_clean_duration_std_
                    output[key]['n_segments']       = self.temp_2.sig_clean_n_segments_
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_1_valid.sig_clean_percentage_
                    output[key]['duration']         = self.temp_1_valid.sig_clean_duration_
                    output[key]['duration_min']     = self.temp_1_valid.sig_clean_duration_min_
                    output[key]['duration_max']     = self.temp_1_valid.sig_clean_duration_max_
                    output[key]['duration_median']  = self.temp_1_valid.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_1_valid.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1_valid.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_1_valid.sig_clean_duration_std_
                    output[key]['n_segments']       = self.temp_1_valid.sig_clean_n_segments_

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_2_valid.sig_clean_percentage_
                    output[key]['duration']         = self.temp_2_valid.sig_clean_duration_
                    output[key]['duration_min']     = self.temp_2_valid.sig_clean_duration_min_
                    output[key]['duration_max']     = self.temp_2_valid.sig_clean_duration_max_
                    output[key]['duration_median']  = self.temp_2_valid.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_2_valid.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2_valid.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_2_valid.sig_clean_duration_std_
                    output[key]['n_segments']       = self.temp_2_valid.sig_clean_n_segments_
            
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp'
                    keys.append(key)
                    output[key]['percentage']       = self.imp_1.sig_clean_percentage_
                    output[key]['duration']         = self.imp_1.sig_clean_duration_
                    output[key]['duration_min']     = self.imp_1.sig_clean_duration_min_
                    output[key]['duration_max']     = self.imp_1.sig_clean_duration_max_
                    output[key]['duration_median']  = self.imp_1.sig_clean_duration_median_
                    output[key]['duration_iqr']     = self.imp_1.sig_clean_duration_iqr_
                    output[key]['duration_mean']    = self.imp_1.sig_clean_duration_mean_
                    output[key]['duration_std']     = self.imp_1.sig_clean_duration_std_
                    output[key]['n_segments']       = self.imp_1.sig_clean_n_segments_
                    
        else:
            
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'acc'
                    keys.append(key)
                    info = self.accx.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']
                    
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    keys.append(key)
                    info = self.breath_1.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']

            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    keys.append(key)
                    info = self.breath_2.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']

            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    keys.append(key)
                    info = self.ecg.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    keys.append(key)
                    info = self.temp_1.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    keys.append(key)
                    info = self.temp_2.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    keys.append(key)
                    info = self.temp_1_valid.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    keys.append(key)
                    info = self.temp_2_valid.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']
                    
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp'
                    keys.append(key)
                    info = self.imp_1.get_stats_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    output[key]['n_segments']       = info['n_segments']
                    
            
        if verbose > 0:
            print('------------------------------------------------------')
            print('USABLE SIGNAL STATS')
            
            if self.accx:
                if not self.accx.is_empty_:
                    print('Acc              %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['acc']['percentage'],
                           (output['acc']['duration']/coef),
                           time_format,
                           (output['acc']['duration_max']/coef),
                           time_format,
                           ))
                    
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    print('Breath 1         %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['breath_1']['percentage'],
                           (output['breath_1']['duration']/coef),
                           time_format,
                           (output['breath_1']['duration_max']/coef),
                           time_format,
                           ))

            if self.breath_2:
                if not self.breath_2.is_empty_:
                    print('Breath 2         %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['breath_2']['percentage'],
                           (output['breath_2']['duration']/coef),
                           time_format,
                           (output['breath_2']['duration_max']/coef),
                           time_format,
                           ))

            if self.ecg:
                if not self.ecg.is_empty_:
                    print('ECG              %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['ecg']['percentage'],
                           (output['ecg']['duration']/coef),
                           time_format,
                           (output['ecg']['duration_max']/coef),
                           time_format,
                           ))

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    print('Temp 1           %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_1']['percentage'],
                           (output['temp_1']['duration']/coef),
                           time_format,
                           (output['temp_1']['duration_max']/coef),
                           time_format,
                           ))

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    print('Temp 2           %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_2']['percentage'],
                           (output['temp_2']['duration']/coef),
                           time_format,
                           (output['temp_2']['duration_max']/coef),
                           time_format,
                           ))
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    print('Temp 1 valid     %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_1_valid']['percentage'],
                           (output['temp_1_valid']['duration']/coef),
                           time_format,
                           (output['temp_1_valid']['duration_max']/coef),
                           time_format,
                           ))

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    print('Temp 2 valid     %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_2_valid']['percentage'],
                           (output['temp_2_valid']['duration']/coef),
                           time_format,
                           (output['temp_2_valid']['duration_max']/coef),
                           time_format,
                           ))
                    
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    print('Imp              %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['imp']['percentage'],
                           (output['imp']['duration']/coef),
                           time_format,
                           (output['imp']['duration_max']/coef),
                           time_format,
                           ))
        
        return output
    
    def get_sig_not_clean_stats(self, from_time=None, to_time=None, time_format='m', verbose=0):

        output = {}
        keys = ['acc', 'breath_1', 'breath_2', 'ecg', 
                'temp_1', 'temp_2', 'temp_1_valid', 'temp_2_valid', 
                'imp',
                'ecg_breath', 'ecg_temp', 'ecg_breath_temp']
        keys2 = ['percentage', 'duration', 'duration_min', 'duration_max',
                 'duration_median', 'duration_iqr', 'duration_mean', 'duration_std']
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = None

        if not self.flag_clean_:
            return output

        if time_format == 's':
            coef = 1
        elif time_format == 'm':
            coef = 60
        elif time_format == 'h':
            coef = 3600
        elif time_format == 'd':
            coef = 3600*24
        else:
            raise NameError('time_format is not correct')
            
        keys = []
        if from_time is None and to_time is None:
            if self.accx:
                if not self.accx.is_empty_:
                    key = 'acc'
                    keys.append(key)
                    output[key]['percentage']       = self.accx.sig_not_clean_percentage_
                    output[key]['duration']         = self.accx.sig_not_clean_duration_
                    output[key]['duration_min']     = self.accx.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.accx.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.accx.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.accx.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.accx.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.accx.sig_not_clean_duration_std_
                    
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    keys.append(key)
                    output[key]['percentage']       = self.breath_1.sig_not_clean_percentage_
                    output[key]['duration']         = self.breath_1.sig_not_clean_duration_
                    output[key]['duration_min']     = self.breath_1.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.breath_1.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.breath_1.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.breath_1.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.breath_1.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.breath_1.sig_not_clean_duration_std_

            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    keys.append(key)
                    output[key]['percentage']       = self.breath_2.sig_not_clean_percentage_
                    output[key]['duration']         = self.breath_2.sig_not_clean_duration_
                    output[key]['duration_min']     = self.breath_2.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.breath_2.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.breath_2.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.breath_2.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.breath_2.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.breath_2.sig_not_clean_duration_std_

            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    keys.append(key)
                    output[key]['percentage']       = self.ecg.sig_not_clean_percentage_
                    output[key]['duration']         = self.ecg.sig_not_clean_duration_
                    output[key]['duration_min']     = self.ecg.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.ecg.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.ecg.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.ecg.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.ecg.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.ecg.sig_not_clean_duration_std_

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_1.sig_not_clean_percentage_
                    output[key]['duration']         = self.temp_1.sig_not_clean_duration_
                    output[key]['duration_min']     = self.temp_1.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.temp_1.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.temp_1.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_1.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_1.sig_not_clean_duration_std_

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_2.sig_not_clean_percentage_
                    output[key]['duration']         = self.temp_2.sig_not_clean_duration_
                    output[key]['duration_min']     = self.temp_2.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.temp_2.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.temp_2.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_2.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_2.sig_not_clean_duration_std_
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_1_valid.sig_not_clean_percentage_
                    output[key]['duration']         = self.temp_1_valid.sig_not_clean_duration_
                    output[key]['duration_min']     = self.temp_1_valid.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.temp_1_valid.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.temp_1_valid.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_1_valid.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_1_valid.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_1_valid.sig_not_clean_duration_std_

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    keys.append(key)
                    output[key]['percentage']       = self.temp_2_valid.sig_not_clean_percentage_
                    output[key]['duration']         = self.temp_2_valid.sig_not_clean_duration_
                    output[key]['duration_min']     = self.temp_2_valid.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.temp_2_valid.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.temp_2_valid.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.temp_2_valid.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.temp_2_valid.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.temp_2_valid.sig_not_clean_duration_std_
                    
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp'
                    keys.append(key)
                    output[key]['percentage']       = self.imp_1.sig_not_clean_percentage_
                    output[key]['duration']         = self.imp_1.sig_not_clean_duration_
                    output[key]['duration_min']     = self.imp_1.sig_not_clean_duration_min_
                    output[key]['duration_max']     = self.imp_1.sig_not_clean_duration_max_
                    output[key]['duration_median']  = self.imp_1.sig_not_clean_duration_median_
                    output[key]['duration_iqr']     = self.imp_1.sig_not_clean_duration_iqr_
                    output[key]['duration_mean']    = self.imp_1.sig_not_clean_duration_mean_
                    output[key]['duration_std']     = self.imp_1.sig_not_clean_duration_std_
                    
        else:

            if self.accx:
                if not self.accx.is_empty_:
                    key = 'acc'
                    keys.append(key)
                    info = self.accx.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    key = 'breath_1'
                    keys.append(key)
                    info = self.breath_1.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']

            if self.breath_2:
                if not self.breath_2.is_empty_:
                    key = 'breath_2'
                    keys.append(key)
                    info = self.breath_2.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']

            if self.ecg:
                if not self.ecg.is_empty_:
                    key = 'ecg'
                    keys.append(key)
                    info = self.ecg.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    key = 'temp_1'
                    keys.append(key)
                    info = self.temp_1.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    key = 'temp_2'
                    keys.append(key)
                    info = self.temp_2.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    key = 'temp_1_valid'
                    keys.append(key)
                    info = self.temp_1_valid.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    key = 'temp_2_valid'
                    keys.append(key)
                    info = self.temp_2_valid.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
            
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    key = 'imp'
                    keys.append(key)
                    info = self.imp_1.get_stats_not_clean_sig_fromto(from_time, to_time)
                    output[key]['percentage']       = info['percentage']
                    output[key]['duration']         = info['duration']
                    output[key]['duration_min']     = info['duration_min']
                    output[key]['duration_max']     = info['duration_max']
                    output[key]['duration_median']  = info['duration_median']
                    output[key]['duration_iqr']     = info['duration_iqr']
                    output[key]['duration_mean']    = info['duration_mean']
                    output[key]['duration_std']     = info['duration_std']
                    
            
        if verbose > 0:
            print('------------------------------------------------------')
            print('ARTIFACTS STATS')
            if self.accx:
                if not self.accx.is_empty_:
                    print('Acc              %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['acc']['percentage'],
                           (output['acc']['duration']/coef),
                           time_format,
                           (output['acc']['duration_max']/coef),
                           time_format,
                           ))
                    
            if self.breath_1:
                if not self.breath_1.is_empty_:
                    print('Breath 1         %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['breath_1']['percentage'],
                           (output['breath_1']['duration']/coef),
                           time_format,
                           (output['breath_1']['duration_max']/coef),
                           time_format,
                           ))

            if self.breath_2:
                if not self.breath_2.is_empty_:
                    print('Breath 2         %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['breath_2']['percentage'],
                           (output['breath_2']['duration']/coef),
                           time_format,
                           (output['breath_2']['duration_max']/coef),
                           time_format,
                           ))

            if self.ecg:
                if not self.ecg.is_empty_:
                    print('ECG              %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['ecg']['percentage'],
                           (output['ecg']['duration']/coef),
                           time_format,
                           (output['ecg']['duration_max']/coef),
                           time_format,
                           ))

            if self.temp_1:
                if not self.temp_1.is_empty_:
                    print('Temp 1           %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_1']['percentage'],
                           (output['temp_1']['duration']/coef),
                           time_format,
                           (output['temp_1']['duration_max']/coef),
                           time_format,
                           ))

            if self.temp_2:
                if not self.temp_2.is_empty_:
                    print('Temp 2           %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_2']['percentage'],
                           (output['temp_2']['duration']/coef),
                           time_format,
                           (output['temp_2']['duration_max']/coef),
                           time_format,
                           ))
                    
            if self.temp_1_valid:
                if not self.temp_1_valid.is_empty_:
                    print('Temp 1 valid   %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_1_valid']['percentage'],
                           (output['temp_1_valid']['duration']/coef),
                           time_format,
                           (output['temp_1_valid']['duration_max']/coef),
                           time_format,
                           ))

            if self.temp_2_valid:
                if not self.temp_2_valid.is_empty_:
                    print('Temp 2 valid %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['temp_2_valid']['percentage'],
                           (output['temp_2_valid']['duration']/coef),
                           time_format,
                           (output['temp_2_valid']['duration_max']/coef),
                           time_format,
                           ))
                    
            if self.imp_1:
                if not self.imp_1.is_empty_:
                    print('Imp              %.2f%% <=> %.2f %s (max %.2f %s )' % 
                          (output['imp']['percentage'],
                           (output['imp']['duration']/coef),
                           time_format,
                           (output['imp']['duration_max']/coef),
                           time_format,
                           ))
        
        return output
    
    def get_main_results(self, from_time=None, to_time=None, verbose=0):

        output = {}
        if not self.flag_analyze_:
            return output
        
        output['acc'] = {}
        output['acc']['n_steps']                = None
        output['acc']['mean_activity']          = None
        output['acc']['mean_activity_level']    = None

        output['breath_1'] = {}
        output['breath_1']['rpm']               = None
        output['breath_1']['rpm_var']           = None
        output['breath_1']['peaks_amps_mv']     = None
        
        output['breath_2'] = {}
        output['breath_2']['rpm']               = None
        output['breath_2']['rpm_var']           = None
        output['breath_2']['peaks_amps_mv']     = None
        
        output['ecg'] = {}
        output['ecg']['bpm']                    = None
        output['ecg']['bpm_var']                = None
        output['ecg']['rr']                     = None
        output['ecg']['hrv']                    = None
        output['ecg']['pnn50']                  = None
        output['ecg']['peaks_amps_mv']          = None
        
        output['temp_1'] = {}
        output['temp_1']['mean']                = None
        output['temp_1']['std']                 = None
        output['temp_1']['var_mean']            = None
        output['temp_1']['var_std']             = None
        
        output['temp_2'] = {}
        output['temp_2']['mean']                = None
        output['temp_2']['std']                 = None
        output['temp_2']['var_mean']            = None
        output['temp_2']['var_std']             = None
        
        if from_time is None and to_time is None:
            if self.accx:
                if not self.accx.is_empty_:
                    output['acc']['n_steps']                = self.accx.n_steps_
                    output['acc']['mean_activity']          = self.accx.mean_activity_
                    output['acc']['mean_activity_level']    = self.accx.mean_activity_level_

            if self.breath_1:
                if not self.breath_1.is_empty_:
                    if len(self.breath_1.rpm_) > 0:
                        output['breath_1']['rpm']           = self.breath_1.rpm_
                        output['breath_1']['rpm_var']       = self.breath_1.rpm_var_
                    
            if self.breath_2:
                if not self.breath_2.is_empty_:

                    if len(self.breath_2.rpm_) > 0:
                        output['breath_2']['rpm']           = self.breath_2.rpm_
                        output['breath_2']['rpm_var']       = self.breath_2.rpm_var_

            if self.ecg:
                if not self.ecg.is_empty_:
                    
                    if len(self.ecg.bpm_) > 0:
                        output['ecg']['bpm']            = self.ecg.bpm_
                        output['ecg']['bpm_var']        = self.ecg.bpm_var_
                        output['ecg']['rr']             = self.ecg.bpm_ms_
                        output['ecg']['hrv']            = self.ecg.bpm_var_ms_
                        output['ecg']['pnn50']          = self.ecg.pnn50_
                        
            if self.temp_1:
                if not self.temp_1.is_empty_:
                    medians     = unwrap(self.temp_1.sig_median_)
                    var_medians = unwrap(self.temp_1.sig_var_median_)
                    if len(medians) > 0:
                        output['temp_1']['mean']        = np.mean(medians)
                        output['temp_1']['std']         = np.std(medians)
                        output['temp_1']['var_mean']    = np.mean(var_medians)
                        output['temp_1']['var_std']     = np.std(var_medians)
            
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    medians     = unwrap(self.temp_2.sig_median_)
                    var_medians = unwrap(self.temp_2.sig_var_median_)
                    if len(medians) > 0:
                        output['temp_2']['mean']        = np.mean(medians)
                        output['temp_2']['std']         = np.std(medians)
                        output['temp_2']['var_mean']    = np.mean(var_medians)
                        output['temp_2']['var_std']     = np.std(var_medians)

        # else:
        #     if self.accx:
        #         if not self.accx.is_empty_:
        #             fs = self.accx.fs_
        #             times, sigx = self.accx.select_on_times(self.accx.times_, self.accx.sig_, from_time, to_time) 
        #             _, sigy     = self.accy.select_on_times(self.accy.times_, self.accy.sig_, from_time, to_time)
        #             _, sigz     = self.accz.select_on_times(self.accz.times_, self.accz.sig_, from_time, to_time)
        #             accx        = Acceleration_x({'times': times, 'sig': sigx, 'fs': fs})
        #             accy        = Acceleration_y({'times': times, 'sig': sigy, 'fs': fs})
        #             accz        = Acceleration_z({'times': times, 'sig': sigz, 'fs': fs})
        #             if not accx.is_empty_:
        #                 accs = Accelerations(accx, accy, accz)
        #                 accs.filt()
        #                 accs.clean()
        #                 accs.analyze()
        #                 output['acc']['n_steps']                = accs.n_steps_
        #                 output['acc']['mean_activity']          = accs.mean_activity_
        #                 output['acc']['mean_activity_level']    = accs.mean_activity_level_

        #     if self.breath_1:
        #         if not self.breath_1.is_empty_:
        #             if is_list_of_list(self.breath_1.rpm_):
        #                 len_rpm = len(unwrap(self.breath_1.rpm_))
        #             else:
        #                 len_rpm = len(self.breath_1.rpm_)
        #             if len_rpm > 0:
        #                 rpm_times       = self.breath_1.rpm_times_
        #                 rpm             = self.breath_1.rpm_
        #                 rpm_var         = self.breath_1.rate_pm_var_
        #                 peaks_amps_mv    = self.breath_1.peaks_amps_mv_ 
        #                 times           = self.breath_1.rate_pm_times_start_
                        
        #                 _, rpm_times_fromto = self.breath_1.select_on_times(times,
        #                                                                     rpm_times,
        #                                                                     from_time,
        #                                                                     to_time)
        #                 _, rpm_fromto = self.breath_1.select_on_times(times,
        #                                                               rpm,
        #                                                               from_time,
        #                                                               to_time)
        #                 _ , rpm_var_fromto = self.breath_1.select_on_times(times,
        #                                                                    rpm_var,
        #                                                                    from_time,
        #                                                                    to_time)
                            
        #                 times           = self.breath_1.peaks_times_
        #                 if len(times) == len(peaks_amps_mv):
                                                                                
        #                      _ , peaks_amps_mv_fromto = self.breath_1.select_on_times(times,
        #                                                                         peaks_amps_mv,
        #                                                                         from_time,
        #                                                                         to_time)
        #                 else:
        #                     peaks_amps_mv_fromto = [0] 
                            
        #                 if len(rpm_fromto) > 0:
        #                     output['breath_1']['rpm']           = rpm
        #                     output['breath_1']['rpm_var']       = rpm_var
        #                     # output['breath_1']['peaks_amps_mv'] = np.median(unwrap(peaks_amps_mv_fromto))

        #     if self.breath_2:
        #         if not self.breath_2.is_empty_:
        #             if is_list_of_list(self.breath_2.rpm_):
        #                 len_rpm = len(unwrap(self.breath_2.rpm_))
        #             else:
        #                 len_rpm = len(self.breath_2.rpm_)
        #             if len_rpm > 0:
        #                 rpm_times       = self.breath_2.rpm_times_
        #                 rpm             = self.breath_2.rate_pm_
        #                 rpm_var         = self.breath_2.rate_pm_var_
        #                 peaks_amps_mv    = self.breath_2.peaks_amps_mv_ 
        #                 times           = self.breath_2.rate_pm_times_start_
        #                 _ , rpm_times_fromto = self.breath_2.select_on_times(times,
        #                                                                      rpm_times,
        #                                                                      from_time,
        #                                                                      to_time)
        #                 _ , rpm_fromto = self.breath_2.select_on_times(times,
        #                                                                     rpm,
        #                                                                     from_time,
        #                                                                     to_time)
        #                 _ , rpm_var_fromto = self.breath_2.select_on_times(times,
        #                                                                    rpm_var,
        #                                                                    from_time,
        #                                                                    to_time)
        #                 times           = self.breath_2.peaks_times_
        #                 if len(times) == len(peaks_amps_mv):
                                                                                
        #                      _ , peaks_amps_mv_fromto = self.breath_2.select_on_times(times,
        #                                                                         peaks_amps_mv,
        #                                                                         from_time,
        #                                                                         to_time)
        #                 else:
        #                     peaks_amps_mv_fromto = [0] 

        #                 if len(rpm_fromto) > 0:
        #                     output['breath_2']['rpm']           = compute_signal_mean(rpm_times_fromto, rpm_fromto)
        #                     output['breath_2']['rpm_var']       = compute_signal_mean(rpm_times_fromto, rpm_var_fromto)
        #                     output['breath_2']['peaks_amps_mv'] = np.median(unwrap(peaks_amps_mv_fromto))

        #     if self.ecg:
        #         if not self.ecg.is_empty_:
        #             if is_list_of_list(self.ecg.bpm_):
        #                 len_bpm = len(unwrap(self.ecg.bpm_))
        #             else:
        #                 len_bpm = len(self.ecg.bpm_)
        #             if len_bpm > 0:
        #                 bpm_times       = self.ecg.bpm_times_
        #                 bpm             = self.ecg.bpm_
        #                 bpm_var         = self.ecg.rate_pm_var_
        #                 rr              = self.ecg.rr_
        #                 hrv             = self.ecg.hrv_
        #                 pnn50           = self.ecg.pnn50_
        #                 peaks_amps_mv   = self.ecg.peaks_amps_mv_ 
        #                 times           = self.ecg.rate_pm_times_start_
        #                 _ , bpm_times_fromto = self.ecg.select_on_times(times,
        #                                                                 bpm_times,
        #                                                                 from_time,
        #                                                                 to_time)
        #                 _ , bpm_fromto = self.ecg.select_on_times(times,
        #                                                           bpm,
        #                                                           from_time,
        #                                                           to_time)
        #                 _ , bpm_var_fromto = self.ecg.select_on_times(times,
        #                                                               bpm_var,
        #                                                               from_time,
        #                                                               to_time)
        #                 _ , rr_fromto = self.ecg.select_on_times(times,
        #                                                          rr,
        #                                                          from_time,
        #                                                          to_time)
        #                 _ , hrv_fromto = self.ecg.select_on_times(times,
        #                                                               hrv,
        #                                                               from_time,
        #                                                               to_time)
                        
        #                 pnn50_fromto = []
        #                 # for i in range(len(times)):
        #                 #     if len(times[i]) > 0:
        #                 #           if times[i][0] > from_time and times[i][0] < to_time:
        #                 #               pnn50_fromto.append(pnn50[0, i])
                                                                                     
        #                 times           = self.ecg.peaks_times_
        #                 _ , peaks_amps_mv_fromto = self.ecg.select_on_times(times,
        #                                                                     peaks_amps_mv,
        #                                                                     from_time,
        #                                                                     to_time)
                            
        #                 if len(bpm_fromto) > 0:
        #                     output['ecg']['bpm']            = compute_signal_mean(bpm_times_fromto, bpm_fromto)
        #                     output['ecg']['bpm_var']        = compute_signal_mean(bpm_times_fromto, bpm_var_fromto)
        #                     output['ecg']['rr']             = 1e3*(60/output['ecg']['bpm'])
        #                     output['ecg']['hrv']            = compute_signal_mean(bpm_times_fromto, hrv_fromto)
        #                     output['ecg']['pnn50']          = np.median(pnn50_fromto)
        #                     output['ecg']['peaks_amps_mv']  = np.median(unwrap(peaks_amps_mv_fromto))

        #     if self.temp_1:
        #         if not self.temp_1.is_empty_:
        #             if is_list_of_list(self.temp_1.sig_median_):
        #                 len_medians = len(unwrap(self.temp_1.sig_median_))
        #             else:
        #                 len_medians = len(self.temp_1.sig_median_)
        #             if len_medians > 0:
        #                 times               = self.temp_1.times_median_
        #                 medians             = self.temp_1.sig_median_
        #                 var_medians         = self.temp_1.sig_var_median_
        #                 _ , medians_fromto  = self.temp_1.select_on_times(times,
        #                                                                   medians,
        #                                                                   from_time,
        #                                                                   to_time)
        #                 _ , var_medians_fromto  = self.temp_1.select_on_times(times,
        #                                                                       var_medians,
        #                                                                       from_time,
        #                                                                       to_time)
                            
        #                 if len(medians_fromto) > 0:
        #                     output['temp_1']['mean']        = np.mean(medians_fromto)
        #                     output['temp_1']['std']         = np.std(medians_fromto)
        #                     output['temp_1']['var_mean']    = np.mean(var_medians_fromto)
        #                     output['temp_1']['var_std']     = np.std(var_medians_fromto)
                                
        #     if self.temp_2:
        #         if not self.temp_2.is_empty_:
        #             if is_list_of_list(self.temp_2.sig_median_):
        #                 len_medians = len(unwrap(self.temp_2.sig_median_))
        #             else:
        #                 len_medians = len(self.temp_2.sig_median_)
        #             if len_medians > 0:
        #                 times               = self.temp_2.times_median_
        #                 medians             = self.temp_2.sig_median_
        #                 var_medians         = self.temp_2.sig_var_median_
        #                 _ , medians_fromto  = self.temp_2.select_on_times(times,
        #                                                                   medians,
        #                                                                   from_time,
        #                                                                   to_time)
        #                 _ , var_medians_fromto  = self.temp_2.select_on_times(times,
        #                                                                       var_medians,
        #                                                                       from_time,
        #                                                                       to_time)
                            
        #                 if len(medians_fromto) > 0:
        #                     output['temp_2']['mean']        = np.mean(medians_fromto)
        #                     output['temp_2']['std']         = np.std(medians_fromto)
        #                     output['temp_2']['var_mean']    = np.mean(var_medians_fromto)
        #                     output['temp_2']['var_std']     = np.std(var_medians_fromto)


        if verbose > 0:
            print('----------------------------------------------------------')
            print('MAIN results')

            if self.accx:
                if not self.accx.is_empty_:
                    print('Number of steps:     ', output['acc']['n_steps'])
                    print('Mean activity:       ', output['acc']['mean_activity'])
                    print('Mean activity level: ', output['acc']['mean_activity_level'])

            if self.breath_1:
                if not self.breath_1.is_empty_:
                    print('RPM thoracic:        ', output['breath_1']['rpm'],
                          '+/-', output['breath_1']['rpm_var'])

            if self.breath_2:
                if not self.breath_2.is_empty_:
                    print('RPM abdominal:       ', output['breath_2']['rpm'],
                          '+/-', output['breath_2']['rpm_var'])

            if self.ecg:
                if not self.ecg.is_empty_:
                    print('BPM:                 ', output['ecg']['bpm'],
                          '+/-', output['ecg']['bpm_var'])
                    print('RR []:               ', output['ecg']['rr'],
                          '+/-', output['ecg']['hrv'])
                    print('pnn50:               ', output['ecg']['pnn50'])
                    
            if self.temp_1:
                if not self.temp_1.is_empty_:
                    print('Temp Right Mean:     ', output['temp_1']['mean'])
                    
            if self.temp_2:
                if not self.temp_2.is_empty_:
                    print('Temp Left Mean:      ', output['temp_2']['mean'])

        return output

    def get_disconnection_time_split(self, from_time, to_time, window_time):

        keys    = ['acc', 'breath', 'ecg', 'temp', 'imp']
        keys2   = ['percentage', 'duration']
        output = {}
        output['from_time']     = None
        output['to_time']       = None
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = None

        tmin = from_time.replace(' ', 'T')
        tmin = np.datetime64(tmin)
        tmax = to_time.replace(' ', 'T')
        tmax = np.datetime64(tmax)

        timestamps = np.arange(tmin, tmax, window_time)
        timestamps = np.append(timestamps, tmax)

        for i in range(len(timestamps)-1):
            from_t  = timestamps[i]
            to_t    = timestamps[i+1]
            results = self.get_disconnections(from_time=from_t, to_time=to_t)

            for key in keys:
                for key2 in keys2:
                    output[key][key2].append(results[key][key2]) 
            output['from_time'].append(from_t)
            output['to_time'].append(to_t)
        
        for key in keys:
                for key2 in keys2:
                    output[key][key2] = np.array(output[key][key2]) 
        
        return output

    def get_sig_clean_stats_time_split(self, from_time, to_time, window_time):

        keys    = ['breath_1', 'breath_2', 'ecg', 
                   'temp_1', 'temp_2', 'temp_1_valid', 'temp_2_valid']
        keys2   = ['percentage', 'duration']
        output = {}
        output['from_time'] = None
        output['to_time']   = None
        for key in keys:
            output[key] = {}
            for key2 in keys2:
                output[key][key2] = None

        tmin = from_time.replace(' ', 'T')
        tmin = np.datetime64(tmin)
        tmax = to_time.replace(' ', 'T')
        tmax = np.datetime64(tmax)

        timestamps = np.arange(tmin, tmax, window_time)
        timestamps = np.append(timestamps, tmax)

        for i in range(len(timestamps)-1):
            from_t  = timestamps[i]
            to_t    = timestamps[i+1]
            results = self.get_sig_clean_stats(from_time=from_t, to_time=to_t)
            output['from_time'].append(from_t)
            output['to_time'].append(to_t)

            for key in keys:
                for key2 in keys2:
                    output[key][key2].append(results[key][key2]) 
        
        for key in keys:
                for key2 in keys2:
                    output[key][key2] = np.array(output[key][key2]) 
        output['timestamps'] = timestamps

        return output

    def get_main_results_time_split(self, from_time, to_time, window_time):

        output                              = {}
        output['from_time']                 = None
        output['to_time']                   = None

        output['acc']                       = {}
        output['acc']['n_steps']            = None

        output['ecg']                       = {}
        output['ecg']['rate_pm']            = None
        output['ecg']['rate_var']           = None
        
        output['breath_1']                  = {}
        output['breath_1']['rate_pm']       = None
        output['breath_1']['rate_pm_var']   = None

        output['breath_2']                  = {}
        output['breath_2']['rate_pm']       = None
        output['breath_2']['rate_pm_var']   = None
        
        output['temp_1']                    = {}
        output['temp_1']['mean']            = None
        output['temp_1']['std']             = None
        output['temp_1']['var_mean']        = None
        output['temp_1']['var_std']         = None
        
        output['temp_2']                    = {}
        output['temp_2']['mean']            = None
        output['temp_2']['std']             = None
        output['temp_2']['var_mean']        = None
        output['temp_2']['var_std']         = None

        tmin = from_time.replace(' ', 'T')
        tmin = np.datetime64(tmin)
        tmax = to_time.replace(' ', 'T')
        tmax = np.datetime64(tmax)

        timestamps = np.arange(tmin, tmax, window_time)
        timestamps = np.append(timestamps, tmax)

        for i in range(len(timestamps)-1):
            from_t = timestamps[i]
            to_t = timestamps[i+1]
            results = self.get_main_results(from_time=from_t, to_time=to_t)
            output['from_time'].append(from_t)
            output['to_time'].append(to_t)

            output['acc']['n_steps'].append(results['acc']['n_steps'])
            output['ecg']['rate_pm'].append(results['ecg']['rate_pm'])
            output['ecg']['rate_var'].append(results['ecg']['rate_var'])
            output['breath_1']['rate_pm'].append(results['breath_1']['rate_pm'])
            output['breath_1']['rate_pm_var'].append(results['breath_1']['rate_pm_var'])
            output['breath_2']['rate_pm'].append(results['breath_2']['rate_pm'])
            output['breath_2']['rate_pm_var'].append(results['breath_2']['rate_pm_var'])
            output['temp_1']['mean'].append(results['temp_1']['mean'])
            output['temp_1']['std'].append(results['temp_1']['std'])
            output['temp_1']['var_mean'].append(results['temp_1']['var_mean'])
            output['temp_1']['var_std'].append(results['temp_1']['var_std'])
            output['temp_2']['mean'].append(results['temp_2']['mean'])
            output['temp_2']['std'].append(results['temp_2']['std'])
            output['temp_2']['var_mean'].append(results['temp_2']['var_mean'])
            output['temp_2']['var_std'].append(results['temp_2']['var_std'])
            
        for key1 in output.keys():
            if key1 != 'from_time' and key1 != 'to_time': 
                for key2 in output[key1].keys():
                    output[key1][key2] = np.array(output[key1][key2])

        output['timestamps'] = timestamps

        return output
    
    
    def savefig_sig(self, path_save, on_sig='filt', flag_acc=False, 
                    flag_breath=False, flag_ecg=False, flag_temp=False, 
                    flag_imp=False):
        
        if not DEV:
            return
        
        plt.close('all')
        sns.set_style('dark')
        if flag_acc:
            self.accx.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'accx_' + on_sig + '.png')
            plt.close('all')
            
            self.accy.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'accy_' + on_sig + '.png')
            plt.close('all')
            
            self.accz.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'accz_' + on_sig + '.png')
            plt.close('all')
        
        if flag_breath:
            self.breath_1.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'breath_1_' + on_sig + '.png')
            plt.close('all')
            
            self.breath_2.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'breath_2_' + on_sig + '.png')
            plt.close('all')
            
        if flag_ecg:
            self.ecg.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'ecg_' + on_sig + '.png')
            plt.close('all')
            
        if flag_temp:
            self.temp_1.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'temp_1_' + on_sig + '.png')
            plt.close('all')
            
            self.temp_2.show(color='C0', on_sig=on_sig)
            plt.savefig(path_save + 'temp_2_' + on_sig + '.png')
            plt.close('all')
            
        if flag_imp:
            self.imp_1.show()
            self.imp_2.show()
            self.imp_3.show()
            self.imp_4.show()
            leg1 = mpatches.Patch(color='C0', label=self.imp_1.name_)
            leg2 = mpatches.Patch(color='C1', label=self.imp_2.name_)
            leg3 = mpatches.Patch(color='C2', label=self.imp_3.name_)
            leg4 = mpatches.Patch(color='C3', label=self.imp_4.name_)
            plt.legend(handles=[leg1, leg2, leg3, leg4], fontsize=11)
            plt.title('Impedance', fontsize=18)
            plt.savefig(path_save + 'imp_' + on_sig + '.png')
            plt.close('all')            
        
                
    def savefig_random_clean_segments(self, path_save, n_seg=6):
        
        if not DEV:
            return
        
        sns.set_style('dark')
        if self.breath_1:
            if not self.breath_1.is_empty_:
                plt.close('all')
                on_sig = 'clean_' + str(self.breath_1.clean_step_)
                self.breath_1.show_random_segments(window=30, on_sig=on_sig, n_seg=n_seg, path_save=path_save)
                plt.close('all')
        
        if self.breath_2:
            if not self.breath_2.is_empty_:
                on_sig = 'clean_' + str(self.breath_2.clean_step_)
                self.breath_2.show_random_segments(window=30, on_sig=on_sig, n_seg=n_seg, path_save=path_save)
                plt.close('all')

        if self.ecg:
            if not self.ecg.is_empty_:
                on_sig = 'clean_' + str(self.ecg.clean_step_)
                self.ecg.show_random_segments(window=7, on_sig=on_sig, n_seg=n_seg, path_save=path_save)
                plt.close('all')
        

    def time_shift(self, offset, time_format='h'):

        if self.accx is not None:
            if not self.accx.is_empty_:
                self.accx.time_shift(offset, time_format)
        if self.accy is not None:
            if not self.accy.is_empty_:
                self.accy.time_shift(offset, time_format)
        if self.accz is not None:
            if not self.accz.is_empty_:
                self.accz.time_shift(offset, time_format)

        if self.breath_1 is not None:
            if not self.breath_1.is_empty_:
                self.breath_1.time_shift(offset, time_format)
        if self.breath_2 is not None:
            if not self.breath_2.is_empty_:
                self.breath_2.time_shift(offset, time_format)

        if self.ecg is not None:
            if not self.ecg.is_empty_:
                self.ecg.time_shift(offset, time_format)

        if self.temp_1 is not None:
            if not self.temp_1.is_empty_:
                self.temp_1.time_shift(offset, time_format)
        if self.temp_2 is not None:
            if not self.temp_2.is_empty_:
                self.temp_2.time_shift(offset, time_format)
                
        if self.temp_1_valid is not None:
            if not self.temp_1_valid.is_empty_:
                self.temp_1_valid.time_shift(offset, time_format)
        if self.temp_2_valid is not None:
            if not self.temp_2_valid.is_empty_:
                self.temp_2_valid.time_shift(offset, time_format)

        if self.imp_1 is not None:
            if not self.imp_1.is_empty_:
                self.imp_1.time_shift(offset, time_format)
            if not self.imp_2.is_empty_:
                self.imp_2.time_shift(offset, time_format)
            if not self.imp_3.is_empty_:
                self.imp_3.time_shift(offset, time_format)
            if not self.imp_4.is_empty_:
                self.imp_4.time_shift(offset, time_format)

            
    def show_random_segments(self, window, on_sig='clean', n_seg=3, path_data=None):
        
        if self.accx:
            if not self.accx.is_empty_:
                self.accx.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)
                
        if self.accy:
            if not self.accy.is_empty_:
                self.accy.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)
                
        if self.accz:
            if not self.accz.is_empty_:
                self.accz.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)
                
        if self.breath_1:
            if not self.breath_1.is_empty_:
                self.breath_1.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)

        if self.breath_2:
            if not self.breath_2.is_empty_:
                self.breath_2.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)

        if self.ecg:
            if not self.ecg.is_empty_:
                self.ecg.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)
        
        if self.temp_1:
            if not self.temp_1.is_empty_:
                self.temp_1.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)
                
        if self.temp_2:
            if not self.temp_2.is_empty_:
                self.temp_2.show_random_segments(window, on_sig=on_sig, n_seg=n_seg, path_data=path_data)
                
         
###############################################################################
###############################################################################
###############################################################################

class Jsonlife(Datalife):
    """ Load, clan, analyze, and show signals from json file downloaded in
    Chronolife's web production application """

    def __init__(self, params):
        """ Constructor

        Parameters
        ----------------
        path:   path json file
        rm_db:  remove double data
                     0: does not remove double data AND raise error
                        if double data
                     1: remove double data
                     2: does not remove double data (Test mode)
        flag_x: flag for loading a given signal type
        from_time, to_time: Min Max of timestamps limit

        """
        self.init_params()
        self.check_params(params)
        check_file(params['path'])
        self.assign_params(params)

    def check_params(self, params):
        """ Check parameters """
        # Missing parameter
        assert 'path' in params.keys(), "path parameter is missing"

    def init_params(self):
        """ Initialize parameters """
        # Init parameters from mother class
        self.init()
        # Properties
        self.path_          = None
        self.user_          = None
        self.datas_json_    = None
        self.rm_db_         = 1
        self.from_time_     = None
        self.to_time_       = None

        # Classes
        self.accx_fromto            = None
        self.accy_fromto            = None
        self.accz_fromto            = None
        self.accs_fromto             = None
        self.breath_1_fromto        = None
        self.breath_2_fromto        = None
        self.ecg_fromto             = None
        self.temp_1_fromto          = None
        self.temp_2_fromto          = None
        self.temp_1_valid_fromto    = None
        self.temp_2_valid_fromto    = None
        self.imp_1_fromto           = None
        self.imp_2_fromto           = None
        self.imp_3_fromto           = None
        self.imp_4_fromto           = None

    def assign_params(self, params):
        """ assign_params parameters

        Parameters
        ----------------
        path:   path json file
        rm_db:  remove double data
                     0: does not remove double data AND raise error
                     if double data
                     1: remove double data
                     2: does not remove double data (Test mode)
        flag_clean:  remove noisy signals & apply filter (boolean)
        flag_x: flag for loading a given signal type
        datas:  json file datas

        """
        # Assign parameters from mother class
        self.assign(params)

        # Assign parameters
        self.path_ = params['path']

        if 'rm_db' in params.keys():
            self.rm_db_ = params['rm_db']

        if 'from_time' in params.keys():
            self.from_time_ = params['from_time']

        if 'to_time' in params.keys():
            self.to_time_ = params['to_time']

    def get_sig(self, signal_type):
        """ Get values and info for a given signal type """

        sig_params = get_sig_info_json(self.path_,
                                       signal_type,
                                       rm_db=self.rm_db_,
                                       rm_dc=True,
                                       verbose=0)

        sig_params['device_model']  = self.device_model_
        sig_params['breath_gain']   = self.breath_gain_
        
        if self.from_time_ is not None or self.to_time_ is not None:
            is_wrapped = is_list_of_list(sig_params['sig'])
            if is_wrapped:
                times, sig, _ = remove_timestamps_unwrap(sig_params['times'],
                                                         sig_params['sig'],
                                                         self.from_time_,
                                                         self.to_time_)
            else:
                times, sig = remove_timestamps(sig_params['times'],
                                               sig_params['sig'],
                                               self.from_time_,
                                               self.to_time_)
            sig_params['times'] = times
            sig_params['sig'] = sig

        return sig_params

    def get_sig_fromto(self, obj, from_time, to_time):

        times   = obj.times_
        sig     = obj.sig_
        fs      = obj.fs_
        times_fromto, sig_fromto = obj.select_on_times(times, sig, from_time,
                                                       to_time)
        sig_params = {'times': times_fromto,
                      'sig': sig_fromto,
                      'fs': fs,
                      'fw_version': self.fw_version_,
                      'device_model': self.device_model_,
                      'breath_gain': self.breath_gain_
                      }

        return sig_params

    def parse(self):
        datas = json_load(self.path_)
        self.user_          = datas['users'][0]['username']
        self.datas_json_    = datas['users'][0]['data']
        self.parse_sig()
        if self.verbose_ > 0:
            self.print_load_errors()


###############################################################################
###############################################################################
###############################################################################

class Apilife(Datalife):
    """ Load and analyze Chronolife data from Cosmos DB """

    def __init__(self, params):
        """ Constructor

        Parameters
        ----------------
        n_jobs: number of cpu for computation
        window_time: Time window for signal segmentation during parallel
        computation (seconds)

        """
        self.init_params()
        self.check_params(params)
        if len(self.errors_) > 0:
            return
        self.assign_params(params)
        self.connect()

    def init_params(self):

        # Init parameters form mother class
        self.init()

        # Init parameters
        self.url_           = None
        self.user_          = None
        self.token_         = None
        self.signal_types_  = None
        self.is_empty_      = False
        self.datas_         = None
        
    def check_params(self, params):

        # Missing parameter
        assert 'path_ids' in params.keys(), 'path_ids parameter is missing'
        assert 'end_user' in params.keys(), 'end_users parameter is missing'
        assert type(params['end_user']) == str, 'end_user should be a string (only 1 end_user accepted)'
        assert 'from_time' in params.keys(), 'from_time parameter is missing'
        assert 'to_time' in params.keys(), 'to_time parameter is missing'
        try:
            start = np.datetime64(params['from_time'])
        except:
            self.errors_.append('Format of "From time" should be: yyyy-mm-dd and hh:mm:ss')
        try:
            stop = np.datetime64(params['to_time'])
        except:
            self.errors_.append('Format of "Date" and "To time" should be: yyyy-mm-dd and hh:mm:ss')
            
        if len(params['end_user']) != 6:
            self.errors_.append('End user id should contain 6 characters')

        listdir = os.listdir(params['path_ids']) 
        filename = 'api_ids.txt'
        if filename not in listdir:
            self.errors_.append('Pylife app and api_ids.txt file should be in the same folder')
            
        if len(self.errors_) > 0:
            return
        
        if start > stop:
            print('Warning! Time start should be anterior to Time stop')

    def assign_params(self, params):

        # Assign parameters form mother class
        self.assign(params)

        self.path_ids_      = params['path_ids']

        if 'api_version' in params.keys():
            self.api_version_ = params['api_version']
        else:
            self.api_version_ = 2
            
        self.utc_offset_    = get_utc_offset(self.time_zone_)
        
        utc_offset_hour     = int(np.floor(self.utc_offset_))
        utc_offset_minute   = int(60*(self.utc_offset_ - np.floor(self.utc_offset_)))
        
        
        from_time       = time_shift_api(params['from_time'], - utc_offset_hour, time_format='h')
        to_time         = time_shift_api(params['to_time'], - utc_offset_hour, time_format='h')
         
        if abs(utc_offset_minute) > 0:
            from_time       = time_shift_api(from_time, - utc_offset_minute, time_format='m')
            to_time         = time_shift_api(to_time, - utc_offset_minute, time_format='m')
             
        # Remove 1 second for a time interval closed [from_time, to_time]
        from_time             = time_shift_api(from_time, - 1, time_format='s')    
        self.from_time_     = from_time
        self.to_time_       = to_time
        self.from_date_     = from_time[:10]
        self.to_date_       = to_time[:10]
        self.date_delta_    = (np.datetime64(self.to_date_) - np.datetime64(self.from_date_))/np.timedelta64(1, 'D')
        self.end_user_      = params['end_user']

        # Signal type to select
        if self.api_version_ == 1:
            signal_types = []
            if self.flag_acc_:
                signal_types.extend(['accx', 'accy', 'accz'])
            if self.flag_breath_:
                signal_types.extend(['breath_1', 'breath_2'])
            if self.flag_ecg_:
                signal_types.extend(['ecg'])
            if self.flag_temp_:
                signal_types.extend(['temp_1', 'temp_2'])
            if self.flag_temp_valid_:
                signal_types.extend(['temp_1_valid', 'temp_2_valid'])
            if self.flag_imp_:
                signal_types.extend(['imp'])
        else:
            signal_types = ''
            if self.flag_acc_:
                signal_types += 'accx,accy,accz,'
            if self.flag_breath_:
                signal_types += 'breath_1,breath_2,'
            if self.flag_ecg_:
                signal_types += 'ecg,'
            if self.flag_temp_:
                signal_types += 'temp_1,temp_2,'
            if self.flag_temp_valid_:
                signal_types += 'temp_1_valid,temp_2_valid,'
            if self.flag_imp_:
                signal_types += 'imp,'
            if signal_types[-1] == ',':
                signal_types = signal_types[:-1]

        self.signal_types_ = signal_types

    def connect(self):
        """ Connection to mongodb server """
        user, token, url = get_ids_api(self.path_ids_)
        # test_login_with_token(url, user, token)

        self.url_   = url
        self.user_  = user
        self.token_ = token

    def get(self):
        """ Load signal information """
        
        # API V1
        if self.api_version_ == 1:
            db_results = get_api(self.user_,
                                  self.token_,
                                  self.url_,
                                  self.end_user_,
                                  self.from_time_,
                                  self.to_time_,
                                  types=self.signal_types_)
            
    
            if len(db_results) == 0:
                self.is_empty_ = True
            else:
                self.datas_ = map_data_api(db_results, self.signal_types_, diagwear=self.diagwear_name_)
    
                mobile_name_s               = []
                diagwear_name_s             = []
                diagwear_firmware_version_s = []
                app_versions_s = []
                for data in self.datas_['users'][0]['data']:
                    mobile_name_s.append(data['mobile_name'])
                    diagwear_name_s.append(data['diagwear_name'])
                    diagwear_firmware_version_s.append(data['diagwear_firmware_version'])
                    app_versions_s.append(data['mobile_app_version'])
    
                self.mobile_names_      = np.unique(mobile_name_s)
                self.diagwear_names_    = np.unique(diagwear_name_s)
                self.fw_versions_       = np.unique(diagwear_firmware_version_s)
                self.app_versions_      = np.unique(app_versions_s)

        else:
            # API V2
            delta = np.datetime64(self.to_date_)-np.datetime64(self.from_date_)
            delta /= np.timedelta64(1, 'D')
                
            dates = []
            from_times = []
            to_times = []
            if delta > 0:
                start = '00:00:00'
                end = '23:59:59'
                for i in range(int(delta+1)):
                    if i > 0:
                        next_day = str(np.datetime64(self.from_date_) + np.timedelta64(i, 'D'))
                        
                    if i == 0:
                        dates.append(self.from_date_)
                        from_times.append(self.from_time_[11:])
                        to_times.append(end)
                            
                    elif i == delta:
                        dates.append(next_day)
                        from_times.append(start)
                        to_times.append(self.to_time_[11:])
                    else:
                        dates.append(next_day)
                        from_times.append(start)
                        to_times.append(end)
                        
            else:
                dates.append(self.from_date_)
                from_times.append(self.from_time_[11:])
                to_times.append(self.to_time_[11:])
                
            
            db_results = []
            for i in range(int(delta+1)):
                db_result = get_api_v2(self.token_,
                                         self.url_,
                                         self.end_user_,
                                         dates[i],
                                         from_times[i],
                                         to_times[i],
                                         types=self.signal_types_)
                db_results.append(db_result)
            db_results = unwrap(db_results)
            
            if len(db_results) == 0:
                self.is_empty_ = True
            else:
                self.datas_ = map_data_api(db_results, self.signal_types_, diagwear=self.diagwear_name_)
    
                mobile_name_s               = []
                diagwear_name_s             = []
                diagwear_firmware_version_s = []
                app_versions_s = []
                for data in self.datas_['users'][0]['data']:
                    mobile_name_s.append(data['mobile_name'])
                    diagwear_name_s.append(data['diagwear_name'])
                    diagwear_firmware_version_s.append(data['diagwear_firmware_version'])
                    app_versions_s.append(data['mobile_app_version'])
    
                self.mobile_names_      = np.unique(mobile_name_s)
                self.diagwear_names_    = np.unique(diagwear_name_s)
                self.fw_versions_       = np.unique(diagwear_firmware_version_s)
                self.app_versions_      = np.unique(app_versions_s)
        
    def get_sig(self, signal_type):
        """ Get values and info for a given signal type """

        if self.is_empty_:
            return []

        output = get_sig_info_api(self.datas_, signal_type, verbose=0)
        output['device_model']  = self.device_model_
        output['ecg_gain']      = self.ecg_gain_
        output['breath_gain']   = self.breath_gain_
        
        return output

    def parse(self):

        if not self.is_empty_:
            self.parse_sig()
            if self.verbose_ > 0:
                self.print_load_errors()    
            utc_offset_hour     = int(np.floor(self.utc_offset_))
            utc_offset_minute   = int(60*(self.utc_offset_ - np.floor(self.utc_offset_)))
            self.time_shift(utc_offset_hour, time_format='h')
            if abs(utc_offset_minute) > 0:
                self.time_shift(utc_offset_minute, time_format='m')

    def compute_on_intervals(self, nb_delta=1, delta='h'):
          
         time_breath_1  = np.array(unwrap(self.breath_1.times_))
         time_breath_2  = np.array(unwrap(self.breath_2.times_))
         time_ecg       = np.array(unwrap(self.ecg.times_))
         time_temp_1    = np.array(unwrap(self.temp_1.times_))
         time_temp_2    = np.array(unwrap(self.temp_2.times_))
         time_x         = np.array(unwrap(self.accx.times_)) 
         time_y         = np.array(unwrap(self.accy.times_))    
         time_z         = np.array(unwrap(self.accz.times_))
         
         indicators_breath_1 = np.array(unwrap(self.breath_1.indicators_clean_2_))
         indicators_breath_2 = np.array(unwrap(self.breath_2.indicators_clean_2_))
         indicators_ecg = np.array(unwrap(self.ecg.indicators_clean_2_))
         indicators_temp_1 = np.array(unwrap(self.temp_1.indicators_))
         indicators_temp_2 = np.array(unwrap(self.temp_2.indicators_))
         acc_x = np.array(unwrap(self.accx.sig_)) 
         acc_y = np.array(unwrap(self.accy.sig_))    
         acc_z = np.array(unwrap(self.accz.sig_))
         
         t = np.min([time_breath_1[0], time_breath_2[0], time_ecg[0],
                    time_temp_1[0], time_temp_2[0]])
         t_end = np.max([time_breath_1[-1], time_breath_2[-1], time_ecg[-1],
                    time_temp_1[-1], time_temp_2[-1]])
         
         resultats_delta = []
         times_delta = []
         
         while t < t_end:
             t_delta_breath_1 = time_breath_1[time_breath_1 > t]
             ind_delta_breath_1 = indicators_breath_1[time_breath_1 > t]        
             t_delta_breath_2 = time_breath_2[time_breath_2 > t]
             ind_delta_breath_2 = indicators_breath_2[time_breath_2 > t]       
             t_delta_ecg = time_ecg[time_ecg > t]
             ind_delta_ecg = indicators_ecg[time_ecg > t]        
             t_delta_temp_1 = time_temp_1[time_temp_1 > t]
             ind_delta_temp_1 = indicators_temp_1[time_temp_1 > t]        
             t_delta_temp_2 = time_temp_2[time_temp_2 > t]
             ind_delta_temp_2 = indicators_temp_2[time_temp_2 > t]
             t_delta_x = time_x[time_x > t]
             ind_delta_x = acc_x[time_x > t]
             t_delta_y = time_x[time_y > t]
             ind_delta_y = acc_y[time_y > t]
             t_delta_z = time_z[time_z > t]
             ind_delta_z = acc_z[time_z > t]
             
             ind_delta_breath_1 = ind_delta_breath_1[t_delta_breath_1 < t + np.timedelta64(nb_delta,
                                                                delta)]
             ind_delta_breath_2 = ind_delta_breath_2[t_delta_breath_2 < t + np.timedelta64(nb_delta,
                                                                delta)]
             ind_delta_ecg = ind_delta_ecg[t_delta_ecg < t + np.timedelta64(nb_delta,
                                                                delta)]
             ind_delta_temp_1 = ind_delta_temp_1[t_delta_temp_1 < t + np.timedelta64(nb_delta,
                                                                delta)]
             ind_delta_temp_2 = ind_delta_temp_2[t_delta_temp_2 < t + np.timedelta64(nb_delta,
                                                                delta)]
             ind_delta_x = ind_delta_x[t_delta_x < t + np.timedelta64(nb_delta,
                                                                delta)]
             ind_delta_y = ind_delta_y[t_delta_y < t + np.timedelta64(nb_delta,
                                                                delta)]
             ind_delta_z = ind_delta_z[t_delta_z < t + np.timedelta64(nb_delta,
                                                                delta)]
         
             t_delta_breath_1 = t_delta_breath_1[t_delta_breath_1 < t + np.timedelta64(nb_delta, delta)]        
             t_delta_breath_2 = t_delta_breath_2[t_delta_breath_2 < t + np.timedelta64(nb_delta, delta)]        
             t_delta_ecg = t_delta_ecg[t_delta_ecg < t + np.timedelta64(nb_delta, delta)]        
             t_delta_temp_1 = t_delta_temp_1[t_delta_temp_1 < t + np.timedelta64(nb_delta, delta)]       
             t_delta_temp_2 = t_delta_temp_2[t_delta_temp_2 < t + np.timedelta64(nb_delta, delta)]
             t_delta_x = t_delta_x[t_delta_x < t + np.timedelta64(nb_delta, delta)]
             t_delta_y = t_delta_y[t_delta_y < t + np.timedelta64(nb_delta, delta)]
             t_delta_z = t_delta_z[t_delta_z < t + np.timedelta64(nb_delta, delta)]
             
             
           
             bin_time_breath_1, bin_breath_1,\
                 loss_b1 = remove_disconnection_loss(t_delta_breath_1,
                                                     ind_delta_breath_1,
                                                     self.breath_1.fs_)
             
             bin_time_breath_2, bin_breath_2,\
                 loss_b2 = remove_disconnection_loss(t_delta_breath_2,
                                                     ind_delta_breath_2,
                                                     self.breath_1.fs_)
             bin_time_ecg, bin_ecg,\
                 loss_ecg = remove_disconnection_loss(t_delta_ecg,
                                                      ind_delta_ecg,
                                                      self.ecg.fs_)
             bin_time_temp_1, bin_temp_1,\
                 loss_temp1 = remove_disconnection_loss(t_delta_temp_1,
                                                        ind_delta_temp_1,
                                                        self.temp_1.fs_)
             bin_time_temp_2, bin_temp_2,\
                 loss_temp2 = remove_disconnection_loss(t_delta_temp_2 ,
                                                        ind_delta_temp_2,
                                                        self.temp_2.fs_)
             bin_time_x, bin_x,\
                 loss_x = remove_disconnection_loss(t_delta_x ,
                                                    ind_delta_x,
                                                    self.accx.fs_)
             
                 
             res_delta = np.zeros((19, ))
             
            
             
             bin_empty = True
             if bin_time_breath_1[0].tolist():
                 unwrap_bin_ind = unwrap(bin_breath_1)
                 #max_clean_ = get_max_length_clean(bin_breath_1, self.breath_1.fs_)
                 median_clean_ = get_median_length_clean(bin_breath_1, self.breath_1.fs_)
                 res_delta[8] = median_clean_
                 res_delta[1] = 100*len(np.where(np.array(unwrap_bin_ind) == 1)[0]
                                           )/len(unwrap_bin_ind)
                 l_deco = 0
                 for i in loss_b1:
                      if i!='NA':
                           l_deco+=i
                 
                 lgth = np.datetime64(t_delta_breath_1[-1])\
                        - np.datetime64(t_delta_breath_1[0])
                 if l_deco==0:
                      res_delta[16] = 0
                 else:
                      res_delta[16] = int(str(l_deco)[:-13]) *100/int(str(lgth)[:-13]) 
                 
                 bin_empty = False
             else:
                 res_delta[8] = -1
                 res_delta[1] = -1
                 res_delta[16] = -1
               
             if bin_time_breath_2[0].tolist():
                 unwrap_bin_ind = unwrap(bin_breath_2)
                 #max_clean_ = get_max_length_clean(bin_breath_2, self.breath_2.fs_)
                 median_clean_ = get_median_length_clean(bin_breath_2, self.breath_2.fs_)
                 res_delta[9] = median_clean_
                 res_delta[2] = 100*len(np.where(np.array(unwrap_bin_ind) == 1)[0]
                                           )/len(unwrap_bin_ind)
                 bin_empty = False
             else:
                 res_delta[9] = -1
                 res_delta[2] = -1
                 
             if bin_time_ecg[0].tolist():
                 unwrap_bin_ind = unwrap(bin_ecg)
                 median_clean_ = get_median_length_clean(bin_ecg, self.ecg.fs_)
              
                 res_delta[7] = median_clean_
                 res_delta[0] = 100*len(np.where(np.array(unwrap_bin_ind) == 1)[0]
                                           )/len(unwrap_bin_ind)
                 
                 l_deco = 0
                 for i in loss_ecg:
                      if i!='NA':
                           l_deco+=i                 
                 lgth = np.datetime64(t_delta_ecg[-1])\
                        - np.datetime64(t_delta_ecg[0]) 
                 if l_deco==0:
                      res_delta[15] = 0
                 else:
                      res_delta[15] = int(str(l_deco)[:-13]) *100/int(str(lgth)[:-13])  
                 
                 bin_empty = False
             else:
                 res_delta[7] = -1
                 res_delta[0] = -1
                 res_delta[15] = -1 
                 
             if bin_time_temp_1[0].tolist():
                 unwrap_bin_ind = unwrap(bin_temp_1)
                 median_clean_ = get_median_length_clean(bin_temp_1, self.temp_1.fs_)
                 res_delta[10] = median_clean_
                 res_delta[3] = 100*len(np.where(np.array(unwrap_bin_ind) == 1)[0]
                                           )/len(unwrap_bin_ind)
                 bin_empty = False
                 l_deco = 0
                 for i in loss_temp1:
                      if i!='NA':
                           l_deco+=i
                 
                 lgth = np.datetime64(t_delta_temp_1[-1])\
                        - np.datetime64(t_delta_temp_1[0])   
                 
                 if l_deco==0:
                      res_delta[17] = 0
                 else:
                      res_delta[17] = int(str(l_deco)[:-13]) *100/int(str(lgth)[:-13])
             else:
                 res_delta[10] = -1
                 res_delta[3] = -1
                 res_delta[17] = -1
                 
             if bin_time_temp_2[0].tolist():
                 unwrap_bin_ind = unwrap(bin_temp_2)
                 median_clean_ = get_median_length_clean(bin_temp_2, self.temp_2.fs_)
                 res_delta[11] = median_clean_
                 res_delta[4] = 100*len(np.where(np.array(unwrap_bin_ind) == 1)[0]
                                           )/len(unwrap_bin_ind)
                 bin_empty = False
                 
             else:
                 res_delta[11] = -1
                 res_delta[4] = -1
                 
                 
             if bin_time_x[0].tolist():
                 
                 activity_level = compute_activity_level_unwrap([ind_delta_x],
                                                                [ind_delta_y],
                                                                [ind_delta_z])
                 mean_activity_level = np.mean(unwrap(activity_level))
                 
                 res_delta[14] = mean_activity_level
                 l_deco = 0
                 for i in loss_x:
                      if i!='NA':
                           l_deco+=i
                 
                 lgth = np.datetime64(t_delta_x[-1])\
                        - np.datetime64(t_delta_x[0])                

                 if l_deco==0:
                      res_delta[18] = 0
                 else:
                      res_delta[18] = int(str(l_deco)[:-13]) *100/int(str(lgth)[:-13])
                 bin_empty = False
             else:
                 
                 res_delta[14] = -1
                 res_delta[18] = -1
             if not bin_empty:
                  resultats_delta.append(res_delta)     
                  times_delta.append(str(t))    
     
                  
             t = t + np.timedelta64(nb_delta, delta)
                
         return times_delta, resultats_delta


    def compute_intersection(self, nb_delta=1, delta='h', all_=True):
          
         time_breath_1  = np.array(unwrap(self.breath_1.times_))
         time_breath_2  = np.array(unwrap(self.breath_2.times_))
         time_ecg       = np.array(unwrap(self.ecg.times_))
         
         indicators_breath_1    = np.array(unwrap(self.breath_1.indicators_clean_2_))
         indicators_breath_2    = np.array(unwrap(self.breath_2.indicators_clean_2_))
         indicators_ecg         = np.array(unwrap(self.ecg.indicators_clean_2_))
        
         id_b1      = np.where(indicators_breath_1==1)[0]
         tb1_clean  = time_breath_1[id_b1]
          
         id_b2      = np.where(indicators_breath_2==1)[0]
         tb2_clean  = time_breath_2[id_b2]
     
         id_ecg     = np.where(indicators_ecg==1)[0]
         tecg_clean = time_ecg[id_ecg]
         
         if all_:
             time_temp_1 = np.array(unwrap(self.temp_1.times_))
             time_temp_2 = np.array(unwrap(self.temp_2.times_))
             
             indicators_temp_1 = np.array(unwrap(self.temp_1.indicators_))
             indicators_temp_2 = np.array(unwrap(self.temp_2.indicators_))
             
             id_t1      = np.where(indicators_temp_1==1)[0]
             tt1_clean  = time_temp_1[id_t1]
               
             id_t2      = np.where(indicators_temp_2==1)[0]
             tt2_clean  = time_temp_2[id_t2]
              
         id_intersect_breath_clean  = np.intersect1d(tb1_clean, tb2_clean)
         id_intersect_clean         = np.intersect1d(id_intersect_breath_clean,
                                                     tecg_clean)
         id_intersect_breath_time   = np.intersect1d(time_breath_1,
                                                     time_breath_2)
         id_intersect_time          = np.intersect1d(id_intersect_breath_time,
                                                     time_ecg)
         
         if all_:
             id_intersect_clean     = np.intersect1d(id_intersect_clean,
                                                     tt1_clean)
             id_intersect_clean     = np.intersect1d(id_intersect_clean,
                                                     tt2_clean)
             id_intersect_time      = np.intersect1d(id_intersect_time,
                                                     time_temp_2) 
         
         t = id_intersect_time[0]
         
         indicators_delta       = []
         times_delta            = []
         max_lenght_delta       = []
         median_lenght_delta    = []
         while t < id_intersect_time[-1]:
             t_delta    = id_intersect_time[id_intersect_time > t]
             ind_delta  = id_intersect_clean[id_intersect_clean > t]
             ind_delta  = ind_delta[ind_delta < t + np.timedelta64(nb_delta, delta)]
             t_delta    = t_delta[t_delta < t + np.timedelta64(nb_delta, delta)]
             
             if t_delta.tolist():
                  indicators_delta.append(100*len(ind_delta)/len(t_delta))
                  fs_ = self.breath_1.fs_
                  if all_:
                      fs_ = self.temp_1.fs_
                 
                  bin_time, bin_ind, _ = remove_disconnection(ind_delta, ind_delta,
                                                              fs_)
                  l = []
                  for i in range(len(bin_time)):
                       l.append(len(bin_time[i]))
                  max_lenght_delta.append(np.max(l)) 
                  median_lenght_delta.append(np.median(l))
                  times_delta.append(t_delta)
             t = t + np.timedelta64(nb_delta, delta)
         return times_delta, indicators_delta, max_lenght_delta,\
                median_lenght_delta
                
    
    def compute_results(self, name_file, nb_delta, delta, new=True,
                        analyze=False):
        
        if not DEV:
            return
        self.filt()
        self.clean()
        
        if analyze:
             self.ecg.analyze_v2(nb_delta=nb_delta, delta=delta)
             self.breath_1.analyze_v2(nb_delta=nb_delta, delta=delta)
             self.breath_2.analyze_v2(nb_delta=nb_delta, delta=delta)
        
        
        times_delta, resultats_delta = self.compute_on_intervals(nb_delta,
                                                                 delta)
        
        times_delta_ecg_breath, indicators_delta_ecg_breath,\
            max_clean_delta_ecg_breath,\
               median_lenght_delta = self.compute_intersection(nb_delta, 
                                                               delta,
                                                               all_=False)
        
        times_delta_all, indicators_delta_all,\
            max_clean_delta_all,\
               median_lenght_all = self.compute_intersection(nb_delta, 
                                                             delta,
                                                             all_=True)
        

        resultats_delta         = np.array(resultats_delta)
        resultats_delta[:, 5]   = np.array(indicators_delta_ecg_breath)
        resultats_delta[:, 6]   = np.array(indicators_delta_all)
        resultats_delta[:, 12]  = np.array(median_lenght_delta)
        resultats_delta[:, 13]  = np.array(median_lenght_all)

        
        resultats_delta = np.hstack((np.array(times_delta).reshape(-1,1),
                                         resultats_delta))
                
        header_ = ['ts', '% usable ecg', '% usable  breath_1', 
                   '% usable  breath_2', '% temp1', '% temp2', 
                   '% ecg/breath_1/breath_2',
                   '% all',
                   'median len ecg', 'median len breath 1', 
                   'median len breath 2',
                   'median len temp 1', 'median len temp 2',
                   'median len ecg/breath_1/breath_2',
                   'median len all',
                   'activity', '%deco ecg', '%deco breath', '%deco temp', 
                   '% deco acc', '% deco imp']
        
        if new: 
             results_ = pd.DataFrame(columns=header_)
        else:
             results_ = pd.read_excel(name_file)
        for i in range(len(resultats_delta)):
            results_ = results_.append(pd.DataFrame([resultats_delta[i,:]],
                                                     columns=header_))
  
        writer = pd.ExcelWriter(name_file+'.xlsx', engine='xlsxwriter')
        results_.to_excel(writer, sheet_name='ecg')
        writer.save()
        
        # if analyze:
        #      with PdfPages(name_file + '.pdf') as pdf:
        #           for r in range(3):
        #                on_sig = 'clean_' + str(self.ecg.clean_step_)
        #                t_clean, ecg_clean, _ = self.ecg.select_on_sig(on_sig)
        #                id_ecg_clean = randint(0, len(ecg_clean))
        #                plt.figure()
        #                plt.plot(t_clean[id_ecg_clean][:self.ecg.fs_*60], 
        #                         ecg_clean[id_ecg_clean][:self.ecg.fs_*60])
        #                plt.title('ECG')
        #                pdf.savefig()
        #                plt.close()
                  
        #                on_sig = 'clean_' + str(self.breath_1.clean_step_)
        #                t_clean, breath_1_clean, _ = self.breath_1.select_on_sig(on_sig)
        #                id_breath_1 = randint(0, len(breath_1_clean))
        #                plt.figure()
        #                plt.plot(t_clean[id_breath_1][:self.breath_1.fs_*120],
        #                         breath_1_clean[id_breath_1][:self.breath_1.fs_*120])
        #                plt.title('Thoracic breathing')
        #                pdf.savefig()
        #                plt.close()
                       
        #                on_sig = 'clean_' + str(self.breath_2.clean_step_)
        #                t_clean, breath_2_clean, _ = self.breath_2.select_on_sig(on_sig)
        #                id_breath_2 = randint(0, len(breath_2_clean))
        #                plt.figure()
        #                plt.plot(t_clean[id_breath_2][:self.breath_2.fs_*120],
        #                         breath_2_clean[id_breath_2][:self.breath_2.fs_*120])
        #                plt.title('Abdominal breathing')
        #                pdf.savefig()
        #                plt.close()
                  
        #           plt.figure()
        #           td = self.ecg.times_delta_
        #           for i in range(1, len(td)):
        #                td[i] = td[i][11:]
        #           plt.bar(td, self.ecg.bpm_delta_)
        #           plt.title('HR (bpm)')      
        #           pdf.savefig()
        #           plt.close()         
        return results_
    
###############################################################################
###############################################################################
###############################################################################

class Simulife(Datalife):
    """ Load and analyze data from simulation """

    def __init__(self, params):
        """ Constructor """
        self.init_params()
        self.check_params(params)
        self.assign_params(params)
        

    def check_params(self, params):

        # Missing parameter
        assert 'path_data' in params.keys(), 'path_data parameter is missing'
        assert 'flag_1seg' in params.keys(), 'flag_1seg parameter is missing'
        assert 'flag_disconnection' in params.keys(), 'flag_disconnection parameter is missing'
        assert 'flag_clean_acc' in params.keys(), 'flag_clean_acc parameter is missing'
        assert 'flag_clean_breath' in params.keys(), 'flag_clean_breath parameter is missing'
        assert 'flag_clean_ecg' in params.keys(), 'flag_clean_ecg parameter is missing'
        assert 'flag_clean_temp' in params.keys(), 'flag_clean_temp parameter is missing'

    def init_params(self):

        # Init parameters form mother class
        self.init()
        self.flag_1seg_ = None
        self.flag_disconnection_ = False
        self.flag_clean_acc_ = False
        self.flag_clean_breath_ = False
        self.flag_clean_ecg_ = False
        self.flag_clean_temp_ = False
        self.is_empty_ = False
        self.datas_ = None

    def assign_params(self, params):

        # Assign parameters form mother class
        self.assign(params)

        self.path_data_ = params['path_data']
        # Signal type to select
        signal_types = []
        if self.flag_acc_:
            signal_types.extend(['accx', 'accy', 'accz'])
        if self.flag_breath_:
            signal_types.extend(['breath'])
        if self.flag_ecg_:
            signal_types.extend(['ecg'])
        if self.flag_temp_:
            signal_types.extend(['temp'])
        if self.flag_imp_:
            signal_types.extend(['imp'])
        
        if 'flag_1seg' in params.keys():
            self.flag_1seg_ = params['flag_1seg']
        if 'flag_disconnection' in params.keys():
            self.flag_disconnection_ = params['flag_disconnection']
        if 'flag_clean_acc' in params.keys():
            self.flag_clean_acc_ = params['flag_clean_acc']
        if 'flag_clean_breath' in params.keys():
            self.flag_clean_breath_ = params['flag_clean_breath']
        if 'flag_clean_ecg' in params.keys():
            self.flag_clean_ecg_ = params['flag_clean_ecg']
        if 'flag_clean_temp' in params.keys():
            self.flag_clean_temp_ = params['flag_clean_temp']

        self.signal_types_ = signal_types

    def read(self):
        savefilename = ['1seg_','disco_', 'acc_', 'breath_', 'ecg_', 'temp']
        if not self.flag_1seg_:
            savefilename[0] = 'no-' + savefilename[0]
        if not self.flag_disconnection_:
            savefilename[1] = 'no-' + savefilename[1]
        if not self.flag_clean_acc_:
            savefilename[2] = 'no-' + savefilename[2]
        if not self.flag_clean_breath_:
            savefilename[3] = 'no-' + savefilename[3]
        if not self.flag_clean_ecg_:
            savefilename[4] = 'no-' + savefilename[4]
        if not self.flag_clean_temp_:
            savefilename[5] = 'no-' + savefilename[5]
        savefilename = ''.join(savefilename)
        savefilename = self.path_data_ + savefilename
                
        with open((savefilename + '.pkl'), 'rb') as file:
            self.datas_ = pickle.load(file)
            
        if len(self.datas_) == 0:
            self.is_empty_ = True
            
    def get_sig(self, signal_type):
        """ Get values and info for a given signal type """

        if self.is_empty_:
            return []

        output = get_sig_info_simul(self.datas_, signal_type)
        output['device_model']  = self.device_model_
        output['breath_gain']   = self.breath_gain_
        
        return output

    def parse_sig(self, from_time=None, to_time=None):
        """ Load signal information """

        if self.flag_acc_:
            self.accx   = Acceleration_x(self.get_sig('accx'))
            self.accy   = Acceleration_y(self.get_sig('accy'))
            self.accz   = Acceleration_z(self.get_sig('accz'))
            self.accs   = Accelerations(self.accx, self.accy, self.accz)

        if self.flag_breath_:
            self.breath_1 = Breath_1(self.get_sig('breath'))
            self.breath_2 = Breath_2(self.get_sig('breath'))

        if self.flag_ecg_:
            self.ecg = ECG(self.get_sig('ecg'))

        if self.flag_temp_:
            self.temp_1 = Temperature_1(self.get_sig('temp'))
            self.temp_2 = Temperature_2(self.get_sig('temp'))

        if self.flag_imp_:
            self.imp_1 = Impedance_1(self.get_sig('imp'))
            self.imp_2 = Impedance_2(self.get_sig('imp'))
            self.imp_3 = Impedance_3(self.get_sig('imp'))
            self.imp_4 = Impedance_4(self.get_sig('imp'))
                
    def parse(self):
        self.read()
        if not self.is_empty_:
            self.parse_sig()
            if self.verbose_ > 0:
                self.print_load_errors()
