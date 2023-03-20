import json
import datetime
import dateutil.parser
from collections import OrderedDict
import numpy as np
from pylife.remove import remove_disconnection
from pylife.useful import get_fs, get_imp_types
from pylife.env import get_env
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
# --- Add imports for PROD and DEV env


def check_file(path):
    """ Check file
    """
    if np.str.lower(path[-4:]) != 'json':
        raise NameError('Input should link to a json file')


def get_sig_info(path, signal_type, rm_db=0, rm_dc=False, show_plot=False,
                 verbose=1):
    """ Get a given signal type
    Parameters
    ----------------
    signal_type:        signal type

    Return
    ----------------
    output: dictionary containing signal information
    times: signal times
    values: signal values
    fs: signal sampling frequency


    """

    output = {}
    fs = get_fs(signal_type)
    
    imp_types = get_imp_types()
    if signal_type in imp_types:
        rm_dc = False
    data = read_data(path,
                       signal_type,
                       fs,
                       rm_db=rm_db,
                       rm_dc=rm_dc,
                       show_plot=False,
                       verbose=verbose)

    sig     = np.array(data['sig'])
    times   = np.array(data['times'])
    
    if signal_type in imp_types and len(sig) > 0:
        sig = np.array(sig)
        new_sig = []
        for values in sig:
            new_sig.append(values[int(signal_type[-1])-1])
        sig = np.array(new_sig)

    output['times']         = times
    output['sig']           = sig
    output['fs']            = fs
    output['fw_version']    = data['fw_version']
    output['app_version']   = data['app_version']
    output['diagwear_name'] = data['diagwear_name']
    
    return output


def get_stats(path, username, start_time, stop_time, warn_dbl_data,
              sig, loss_):
    stat_ = []
    stat_.append(path[:-5])
    stat_.append(username)
    stat_.append(start_time)
    stat_.append(stop_time)
    stat_.append(warn_dbl_data)
    stat_.append(len(sig))
    stat_.append(loss_)

    return stat_


def json_load(path):
    with open(path, "r") as json_file:
        datas = json.load(json_file)
    if not datas['users'][0]['data']:
        raise NameError('No data')

    return datas


def missing_data(timeseries, signal_type, fs, rm_db=0, verbose=1):
    """ Evaluate disconnection
            Input: timeseries
            Output : loss_, warning_double_data
            loss_ : % of loss due to disconnection
            warning_double_data : if True error data acquisition
            rm_db : remove double data
                    0: does not remove double data and raise error
                    1: remove double data
                    2: does not remove double data (Test mode)
    """
    warning_double_data = False
    ordered_timeseries = OrderedDict(sorted(timeseries.items(),
                                            key=lambda t: t[0]))
    imp_types = get_imp_types()
    if signal_type in imp_types:
        signal_type = 'imp'
    count = []
    date = []
    duration = []
    timestamp = []
    sample_number = []
    for timeserie in ordered_timeseries:
        if signal_type in timeseries[timeserie].keys():
            if signal_type != 'imp':
                count.append(dateutil.parser.parse(timeserie) +
                             datetime.timedelta(seconds=float(len(timeseries[timeserie][signal_type])
                                                              / fs)))
                date.append(dateutil.parser.parse(timeserie))
                duration.append((len(timeseries[timeserie][signal_type]) / fs))
                sample_number.append(len(timeseries[timeserie][signal_type]))
                timestamp.append(timeserie)
                last = timeserie
            else:
                count.append(dateutil.parser.parse(timeserie) +
                             datetime.timedelta(seconds=10))
                date.append(dateutil.parser.parse(timeserie))
                duration.append(1/fs)
                sample_number.append(1)
                timestamp.append(timeserie)
                last = timeserie

    if len(date) > 1:
        diff = (np.array(date)[1:]-np.array(count)[0:-1])/datetime.timedelta(seconds=1)
        # disconnection_durations =  diff*3600*24 # seconds
        if sum(diff < 0) > 0:
            warning_double_data = True
            if rm_db == 1:
                timeseries, ordered_timeseries = remove_double_data(signal_type,
                                                                    timeseries,
                                                                    date,
                                                                    count,
                                                                    duration)
                count = []
                date = []
                duration = []
                timestamp = []
                sample_number = []
                for timeserie in ordered_timeseries:
                    if signal_type in timeseries[timeserie]:
                        count.append(dateutil.parser.parse(timeserie) +
                                     datetime.timedelta(seconds=float(len(timeseries[timeserie][signal_type])
                                                                      / fs)))
                        date.append(dateutil.parser.parse(timeserie))
                        duration.append((len(timeseries[timeserie][signal_type]) / fs))
                        sample_number.append(len(timeseries[timeserie][signal_type]))
                        timestamp.append(timeserie)
                        last = timeserie
                diff = (np.array(date)[1:]
                        - np.array(count)[:-1]) / datetime.timedelta(1, 1)
                # disconnection_durations =  diff*3600*24 # seconds

                print('DOUBLE DATA REMOVED')
            elif rm_db == 0:
                date = np.array(date)
                raise NameError('ERROR: DOUBLE DATA')
            else:
                print('!!! WARNING !!! MODE TEST: DOUBLE DATA KEPT')

    if len(date) > 1:
        diff_ = np.sum(np.array(date)[1:]-np.array(count)[0:-1])
        len_ = datetime.timedelta(seconds=float(len(timeseries[last][signal_type])
                                                / fs))
        loss_ = 100 * diff_ / (date[-1] + len_ - date[0])
        if verbose > 0:
            print('Missing data', signal_type, '=> Loss duration:', diff_,
                  ';  Loss proportion:', round(loss_*10)/10, '%; Time interval',
                  (date[-1]+len_-date[0]))

    else:
        if verbose > 0:
            print('One timestamp for ', signal_type, ' : No missing data')
        loss_ = 0

    return timeseries, loss_, warning_double_data


def parse_data(datas, signal_type, fs, rm_db=0, rm_dc=False,
               show_plot=False, verbose=1):
    """ Get a given signal type
    Parameters
    ----------------
    path :          path json file
    signal_type:    signal type
    fs:             sampling frequency
    rm_db :         remove double data
                            0: does not remove double data and raise error
                            1: remove double data
                            2: does not remove double data (Test mode)
    rm_dc:          remove disconnection (boolean)

    Return
    ----------------
    times, sig, stats, is_empty

    """

    is_empty = False
    data = parse_json(datas,
                      signal_type,
                      fs,
                      show_plot,
                      rm_db=rm_db,
                      verbose=verbose)
    
    # TBD: first argument was the file path, but we don't have it now that data
    # is already loaded. Was it really useful? If so it could be added later.
    stat_ = get_stats("", data['username'], data['start'], data['stop'],
                      data['warning_double_data'], data['sig'], data['loss'])

    if len(data['sig']) < 1:
        is_empty = True

    sig     = np.array(data['sig'])
    times   = np.array(data['times'])
    
    if rm_dc and not is_empty:
        times, sig, stat_ = remove_disconnection(times, sig, 
                                                 fs, stat_)

    output = {}
    output['times']                 = times
    output['sig']                   = sig
    output['is_empty']              = is_empty
    output['username']              = data['username']
    output['loss']                  = data['loss']
    output['start']                 = data['start']
    output['stop']                  = data['stop']
    output['warning_double_data']   = data['warning_double_data']
    output['fw_version']            = data['fw_version']
    output['app_version']           = data['app_version']
    output['diagwear_name']         = data['diagwear_name']
    
    return output


def read_data(path, signal_type, fs, rm_db=0, rm_dc=False,
              show_plot=False, verbose=1):
    with open(path, "r") as file_json:
        return parse_data(
            json.load(file_json),
            signal_type,
            fs,
            rm_db,
            rm_dc,
            show_plot,
            verbose=verbose)


def parse_json(datas, signal_type, fs, show_plot=False, rm_db=0, verbose=1):
    """ Read json data
    Call the missing_data function to compute the number of disconnection.
    Input: path, name_signal, fs, show_plot
    path : path json file
    show_plot : if True plot, plot the physiological signals
    rm_db : remove double data
                    0: does not remove double data and raise error
                    1: remove double data
                    2: does not remove double data (Test mode)
    Output : [time, data]
    """

    if len(datas['users'][0]['data']) == 0:
        raise NameError('No data has been found in "datas" dictionnary')
    timeseries      = {}
    fw_version      = []
    app_version     = []
    diagwear_name   = []
    
    for data in datas['users'][0]['data']:
        if data["timestamp"] not in timeseries.keys():
            timeseries[data["timestamp"]] = {}
        if data["type"] not in timeseries[data["timestamp"]].keys():
            timeseries[data["timestamp"]][data["type"]] = {}
        timeseries[data["timestamp"]][data["type"]] = data["values"]
        if 'diagwear_firmware_version' in data.keys():
            fw_version.append(data['diagwear_firmware_version'])
        elif 'bversion' in data.keys():
            fw_version.append(data['bversion'])
        if 'mobile_app_version' in data.keys():
            app_version.append(data['mobile_app_version'])
        elif 'version' in data.keys():
            app_version.append(data['version'])
        if 'diagwear_name' in data.keys():
            diagwear_name.append(data['diagwear_name'])
        elif 'bdevice' in data.keys():
            diagwear_name.append(data['bdevice'])
    ordered_timeseries = OrderedDict(sorted(timeseries.items(),
                                            key=lambda t: t[0]))
    t = []
    for timeserie in ordered_timeseries:
        t.append(timeserie)
    timeseries, loss_, warning_double_data = missing_data(timeseries,
                                                          signal_type,
                                                          fs,
                                                          rm_db=rm_db,
                                                          verbose=verbose)
    ordered_timeseries = OrderedDict(sorted(timeseries.items(),
                                            key=lambda t: t[0]))
    imp_types = get_imp_types()
    if signal_type in imp_types:
        signal_type = 'imp'

    time        = []
    sig         = []
    
    for timeserie in ordered_timeseries:
        if signal_type in ordered_timeseries[timeserie]:
            if signal_type != 'imp':
                time.extend(np.arange(dateutil.parser.parse(timeserie),
                                      dateutil.parser.parse(timeserie)
                                      + datetime.timedelta(
                                          seconds=float(len(timeseries[timeserie][signal_type]))
                                          / fs), datetime.timedelta(seconds=1 / fs)))
                sig.extend(timeseries[timeserie][signal_type])
                
            else:
                time.extend(np.arange(dateutil.parser.parse(timeserie),
                                      dateutil.parser.parse(timeserie)
                                      + datetime.timedelta(seconds=float(1/fs)),
                                      datetime.timedelta(seconds=1 / fs)))
                sig.append(timeseries[timeserie][signal_type])

    if len(sig) < 1:
        if verbose > 1:
            print('!!! WARNING !!!', signal_type, 'signal is empty')

    if show_plot:
        plt.figure()
        plt.plot(time, sig)
        plt.title(signal_type)

    output = {}
    output['times']                 = time
    output['sig']                   = sig
    output['username']              = datas['users'][0]['username']
    output['loss']                  = loss_
    output['start']                 = t[0]
    output['stop']                  = t[-1]
    output['warning_double_data']   = warning_double_data
    output['fw_version']            = np.unique(fw_version)
    output['app_version']           = np.unique(app_version)
    output['diagwear_name']         = np.unique(diagwear_name)
    
    return output


def read_json(path, signal_type, fs, show_plot=False, rm_db=0, verbose=1):
    """ Read json file
    Call the missing_data function to compute the number of disconnection.
    Input: path, name_signal, fs, show_plot
    path : path json file
    show_plot : if True plot, plot the physiological signals
    rm_db : remove double data
                    0: does not remove double data and raise error
                    1: remove double data
                    2: does not remove double data (Test mode)
    Output : [time, data]
    """
    with open(path, "r") as file_json:
        return parse_json(
            json.load(file_json),
            signal_type,
            fs,
            show_plot,
            rm_db,
            verbose=verbose)


def read_json_all(path, show_plot=True):
    """ Read json file
    Call the missing_data function to compute the number of disconnection.
    Input: path, timestamp1, timestamp2, show_plot
    path : path json file
    timestamp1 : Read data posterior to timestamp1
    timestamp2 : Read data anterior to timestamp2
    show_plot : if True plot, plot the physiological signals

    """
    signal_types = ['ecg', 'breath_1', 'breath_2', 'accx', 'accy', 'accz', 'temp_1', 'temp_2']
    for signal_type in signal_types:
        read_json(path, signal_type, get_fs(signal_type), show_plot)

    return 0


def read_json_wrap(path, signal_type, fs, show_plot=False, rm_db=0):
    """ Read json file
    Call the missing_data function to compute the number of disconnection.
    Input: path, name_signal, fs, show_plot
    path : path json file
    show_plot : if True plot, plot the physiological signals
    rm_db : remove double data
                    0: does not remove double data and raise error
                    1: remove double data
                    2: does not remove double data (Test mode)
    Output : [time, data]
    """
    with open(path, "r") as file_json:
        datas = json.load(file_json)
    if not datas['users'][0]['data']:
        raise NameError('No data')
    timeseries = {}
    for data in datas['users'][0]['data']:
        if data["timestamp"] not in timeseries.keys():
            timeseries[data["timestamp"]] = {}
        if data["type"] not in timeseries[data["timestamp"]].keys():
            timeseries[data["timestamp"]][data["type"]] = {}
        timeseries[data["timestamp"]][data["type"]] = data["values"]
    ordered_timeseries = OrderedDict(sorted(timeseries.items(), key=lambda t: t[0]))
    t = []
    for timeserie in ordered_timeseries:
        t.append(timeserie)
    timeseries, loss_, warning_double_data = missing_data(timeseries,
                                                          signal_type,
                                                          fs,
                                                          rm_db=rm_db)
    ordered_timeseries = OrderedDict(sorted(timeseries.items(), key=lambda t: t[0]))
    time = []
    sig = []
    for timeserie in ordered_timeseries:
        if signal_type in ordered_timeseries[timeserie]:
            time.append(np.arange(dateutil.parser.parse(timeserie),
                                  dateutil.parser.parse(timeserie)
                                  + datetime.timedelta(
                                      seconds=float(len(timeseries[timeserie][signal_type]))
                                      / fs), datetime.timedelta(seconds=1/fs)))
            sig.append(timeseries[timeserie][signal_type])
    if show_plot:
        plt.figure()
        plt.plot(time, sig)
        plt.title(signal_type)

    return [time, sig], datas['users'][0]['username'], loss_, t[0], t[-1], warning_double_data


def remove_double_data(signal_type, timeseries, date, count, duration):
    """ Remove double data and keep the higher signal duration only

    Parameter
    ----------
    name_signal
    timeseries
    date
    count
    duration

    Return
    ----------
    timeseries
    ordered_timeseries
    date
    count
    duration
    last
    """

    count = np.array(count)
    date = np.array(date)
    duration = np.array(duration)
    ordered_timeseries = OrderedDict(sorted(timeseries.items(),
                                            key=lambda t: t[0]))

    diff = (date[1:] - count[0:-1]) / datetime.timedelta(1, 1)
    idouble_s = np.argwhere(diff < 0)[:, 0]
    ikeep = np.arange(len(date))

    idelete_s = []
    for idouble in idouble_s:
        diff_single_double = (date[idouble+1:] - count[idouble])\
                             / datetime.timedelta(1, 1)
        isingle_doubles = idouble\
                        + 1 + np.argwhere(diff_single_double < 0)[:, 0]
        idouble_all = np.append(idouble, isingle_doubles)
        # Keep the higher signal duration among double data
        iduration_max = np.argmax(np.array(duration)[idouble_all])
        idelete = np.delete(idouble_all, iduration_max)
        idelete_s.extend(idelete)

    ikeep = np.delete(ikeep, idelete_s)

    # Update dictionaies
    timeseries_keep = timeseries.copy()
    ordered_timeseries_keep = ordered_timeseries.copy()
    icount = 0
    for timeserie in ordered_timeseries:
        if signal_type in timeseries[timeserie]:
            if icount not in ikeep:
                del timeseries_keep[timeserie]
                del ordered_timeseries_keep[timeserie]
            icount += 1

    timeseries = timeseries_keep.copy()
    ordered_timeseries = ordered_timeseries_keep.copy()

    return timeseries, ordered_timeseries


def extract_sample_json_file(path, name_sig, sampling_frequency, start_samp,
                             end_samp):
    '''
    Divide the json according to the date of the recording
    Inputs: path, name_sig, sampling_frequency, nb_time_delta, time_delta
    path = json file
    name_sig: 'ecg', 'breath_1', 'breath_2', 'accx', 'accy', 'accz', temp_1',
    'temp_2'
    sampling_frequency
    nb_time_delta : (int) number of time_delta
    time_delta : 'D', 'm'...
    Outputs: bin_time, bin_sig
    bin_time: list of timestamps
    bin_sig: list of signals
    '''
    start_samp = np.datetime64(start_samp)
    end_samp = np.datetime64(end_samp)
    [time, sig], username, loss_, start, end,\
        warning_double_data, fw_version = read_json(path, name_sig, sampling_frequency,
                                        show_plot=False)
    time = np.array(time)
    sig = np.array(sig)

    associated_time = time[time >= start_samp]
    associated_sig = sig[time >= start_samp]

    associated_sig = associated_sig[associated_time <= end_samp]
    associated_time = associated_time[associated_time <= end_samp]

    return np.array(associated_time), np.array(associated_sig)
