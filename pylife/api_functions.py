from pylife.env import get_env
DEV = get_env()

import json
import numpy as np
from pylife.useful import get_fs
from pylife.useful import get_imp_types
from pylife.useful import get_signal_types
from pylife.useful import get_signal_filtered_types
from pylife.useful import get_signal_result_types
from pylife.useful import get_app_info_types
from pylife.useful import get_all_api_types
from pylife.parse import parse_data

if DEV:
    import requests
    import warnings
    warnings.filterwarnings("ignore")


def test_login_with_password(user, password, otp_token):
    """Login connection test with user password

    Parameters
    ----------------
    user: user name
    password: user password
    otp_token: double identification
    """

    _ = get_token(user, password, otp_token, verbose=1)


def get_token(url, user, password, otp_token, verbose=1):
    """Get new token for api request

    Parameters
    ----------------
    url: API URL
    user: user name
    password: user password
    otp_token: double identification

    Returns
    ----------------
    token
    """

    # url = 'https://preprod.chronolife.net/api/1/user/check'
    json_request = {
        "device": "Chronolife",
        "otp_token": otp_token
    }

    r = requests.post(url, auth=(user, password), json=json_request)

    r_text = json.loads(r.text)
    if verbose > 0:
        print('status code:', r.status_code)
        print('status:', r_text['status'])
        print('token:', r_text['token'])

    return r_text['token']


def test_login_with_token(url, user, token):
    """ Login connection test with user's token

    Parameters
    ----------------
    url: API URL
    user: user name
    token: token

    """

    # url = 'https://preprod.chronolife.net/api/1/user/events'
    json_request = []
    r = requests.post(url, auth=(user, token), json=json_request)
    print(get_status_message(r.status_code))


def get_ids(path_ids):
    """ "Get ids for api from file for database connection

    Parameters
    ----------------
    path_ids: folder name to read api_ids.txt file


    Returns
    ----------------
    user: user name
    token: token
    url: API URL

    """

    user_key = 'user='
    token_key = 'token='
    url_key = 'url='
    file = 'api_ids.txt'
    f = open(path_ids + "/" + file, "r")
    line = 0
    while line != '':
        line = f.readline()
        line = line.replace(' ', '')
        line = line.replace('\n', '')
        if user_key in line:
            user = line[len(user_key):]
        elif token_key in line:
            token = line[len(token_key):]
        elif url_key in line:
            url = line[len(url_key):]
    f.close()

    if user == '':
        raise NameError('user seems empty in ' + file)
    if token == '':
        raise NameError('token seems empty in ' + file)
    if url == '':
        raise NameError('url seems empty in ' + file)

    return user, token, url


def get(user, token, url, end_users, from_time, to_time, types=None,
         verbose=1):
    """ get data in database with query using API

    Parameters
    ----------------
    user: user name
    token: token
    url: API URL
    end_users: end_users ids
    from_time: Time min to request
    to_time: : Time max to request
    types: Signal types to request

    Returns
    ----------------
    datas

    """
    datas = []

    if type(end_users) is not list:
        end_users = [end_users]

    from_time = from_time + '.00'
    to_time = to_time + '.00'

    json_request = {
        "gt": from_time,
        "lt": to_time,
        "users": end_users
    }

    if types is not None:
        if type(types) is not list:
            types = [types]
        json_request['types'] = types
    
    r = requests.get(url, auth=(user, token), json=json_request)
    print(url)
    print(user)
    print(token)
    print(json_request)
    if r.status_code == 200:
        r_text = json.loads(r.text)
        datas = r_text['data']
    else:
        print('------------------------------------------')
        print(get_status_message(r.status_code))
        return []

    while r_text['offset'] > 0:

        # if verbose > 0:
        #     print('Number of events', len(datas), 'Offset', r_text['offset'])

        json_request['offset'] = r_text['offset']
        r = requests.get(url, auth=(user, token), json=json_request)
        if r.status_code == 200:
            r_text = json.loads(r.text)
            for data in r_text['data']:
                datas.append(data)
        else:
            print('------------------------------------------')
            print(get_status_message(r.status_code))

    if len(datas) == 0:
        print('Datas is empty')
    else:
        if verbose > 0:
            print('Number of events total', len(datas), 'Offset',
                  r_text['offset'])

    return datas


def map_data(datas, signal_types, diagwear=None):
    """ Map data to build Jsonprod like architecture

    Parameters
    ----------------
    datas: datas returned by API request returned by get function
    types: Signal types to request

    Returns
    ----------------
    datas mapped

    """
    
    if len(datas) == 0:
        return []

    if signal_types is None:
       signal_types = get_all_api_types()
       
    signal_types_ref = get_signal_types()
    datas_mapped = {}
    datas_mapped['users'] = []
    usernames = []
    types = []
    for data in datas:
        usernames.append(data['user'])
        types.append(data['type'])

    for username in np.unique(usernames):
        data_user = []
        for data in datas:
            if username == data['user'] and data['type'] in signal_types and\
                    data['type'] in signal_types_ref:
                
                if diagwear is not None:
                    if diagwear != data['bdevice']:
                        continue
                    
                values = data['values']
                if type(values) is not list:
                    values = [data['values']]

                data_user.append({'_id':                        data['_id'],
                                  'mobile_name':                data['device'],
                                  'diagwear_name':              data['bdevice'],
                                  'diagwear_firmware_version':  data['bversion'],
                                  'mobile_app_version':         data['dversion'],
                                  'type':                       data['type'],
                                  'timestamp':                  data['mtimestamp'],
                                  'values':                     values,
                                  'frequency':                  get_fs(data['type']),
                                  })

        datas_mapped['users'].append({'username': username, 'data': data_user})

    return datas_mapped


def map_data_filt(datas, signal_types):
    """ Map data to build Jsonprod like architecture

    Parameters
    ----------------
    datas: datas returned by API request returned by get function
    types: Signal types to request

    Returns
    ----------------
    datas mapped

    """

    if len(datas) == 0:
        return []
    
    if signal_types is None:
       signal_types = get_all_api_types()

    signal_types_ref = get_signal_filtered_types()
    datas_mapped = {}
    datas_mapped['users'] = []
    usernames = []
    types = []
    for data in datas:
        usernames.append(data['user'])
        types.append(data['type'])

    for username in np.unique(usernames):
        data_user = []
        for data in datas:
            if username == data['user'] and\
                data['type'] in signal_types and\
                    data['type'] in signal_types_ref:
                values = data['values']
                if type(values) is not list:
                    values = [data['values']]

                data_user.append({'_id':                        data['_id'],
                                  'type':                       data['type'],
                                  'timestamp':                  data['mtimestamp'],
                                  'values':                     values,
                                  'frequency':                  get_fs(data['type']),
                                  })

        datas_mapped['users'].append({'username': username, 'data': data_user})


    return datas_mapped


def map_results(datas, result_types):
    """ Map results to build Jsonprod like architecture

    Parameters
    ----------------
    datas: datas returned by API request returned by get function
    types: Result types to request

    Returns
    ----------------
    datas mapped

    """

    if len(datas) == 0:
        return []
    
    if result_types is None:
       result_types = get_all_api_types()

    signal_types_ref = get_signal_result_types()
    datas_mapped = {}
    datas_mapped['users'] = []
    usernames = []
    types = []
    for data in datas:
        usernames.append(data['user'])
        types.append(data['type'])

    for username in np.unique(usernames):
        data_user = []
        for data in datas:
            if username == data['user'] and\
                data['type'] in result_types and\
                    data['type'] in signal_types_ref:
                values = data['values']
                if type(values) is not list:
                    values = [data['values']]

                data_user.append({'_id':                        data['_id'],
                                  'type':                       data['type'],
                                  'timestamp':                  data['mtimestamp'],
                                  'values':                     values,
                                  })

        datas_mapped['users'].append({'username': username, 'data': data_user})

    return datas_mapped


def map_data_app(datas, app_info_types):
    """ Map results to build Jsonprod like architecture

    Parameters
    ----------------
    datas: datas returned by API request returned by get function
    types: Result types to request

    Returns
    ----------------
    datas mapped

    """

    if len(datas) == 0:
        return []
    
    if app_info_types is None:
       app_info_types = get_all_api_types()

    types_ref = get_app_info_types()
    datas_mapped = {}
    datas_mapped['users'] = []
    usernames = []
    types = []
    for data in datas:
        usernames.append(data['user'])
        types.append(data['type'])

    for username in np.unique(usernames):
        data_user = []
        for data in datas:
            if username == data['user'] and\
                data['type'] in app_info_types and\
                    data['type'] in types_ref:
                values = data['values']
                if type(values) is not list:
                    values = [data['values']]

                data_user.append({'_id':                        data['_id'],
                                  'mobile_name':                data['device'],
                                  'diagwear_name':              data['bdevice'],
                                  'diagwear_firmware_version':  data['bversion'],
                                  'mobile_app_version':         data['dversion'],
                                  'type':                       data['type'],
                                  'timestamp':                  data['mtimestamp'],
                                  'values':                     values,
                                  'frequency':                  0,
                                  })

        datas_mapped['users'].append({'username': username, 'data': data_user})

    return datas_mapped


def get_sig_info(datas, signal_type, rm_db=0, rm_dc=True, verbose=0):
    """ Get a given signal type (only for raw data)
    Parameters
    ----------------
    datas:              datas from map_data function
    signal_type:        signal type
    rm_db:              remove double data
    rm_dc:              remove disconnections
    verbose:            display info
    
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
    data = parse_data(datas,
                      signal_type,
                      fs,
                      rm_db=rm_db,
                      rm_dc=rm_dc,
                      show_plot=False,
                      verbose=verbose)

    sig     = data['sig']
    times   = data['times']
    
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


# def get_sig_filt_info(datas, signal_type):
#     """ Get a given signal type (only for signal filtered data)
#     Parameters
#     ----------------
#     datas:              datas from map_data_filtered function
#     result_type:        result type
    
#     Return
#     ----------------
#     output: dictionary containing signal information
#     times: signal times
#     values: signal values
#     fs: signal sampling frequency
#     """
#     output = {}
    
#     datas = datas['users'][0]['data']
#     types = []
#     values = []
#     times = []
#     for data in datas:
#         # types.append(data['type'])
#         if data['type'] == signal_type:
#             values.extend(data['values'])
#             fs = data['frequency']
#             from_time = np.datetime64(data['timestamp'])
#             to_time = from_time + np.timedelta64(int(len(values)/fs*1e6), 'us')
#             step_time = np.timedelta64(int(1/fs*1e6), 'us')
#             times.extend(np.arange(from_time, to_time, step_time))
    
#     frequencies = 1/((times[1:] - times[:-1])/np.timedelta64(1, 's'))
#     frequencies = frequencies.astype('int')
#     wrong_freq = np.where(frequencies != fs)
    
#     if len(wrong_freq[0]) != 0:
#         raise NameError('Signal filtered timestamp is wrong')

#     output['times'] = times
#     output['sig'] = values
#     output['fs'] = fs

#     return output
        
def get_result_info(datas, result_type):
    """ Get a given signal type (only for result data)
    Parameters
    ----------------
    datas:              datas from map_results function
    result_type:        result type
    
    Return
    ----------------
    output: dictionary containing signal information
    times: signal times
    values: signal values
    """
    output = {}
    
    datas = datas['users'][0]['data']
    times = []
    values = []
    for data in datas:
        if result_type == data['type']:
            times.append(np.datetime64(data['timestamp']))
            values.append(data['values'][0:])

    output['times'] = times
    output['values'] = values

    return output


def get_app_info(datas, result_type):
    """ Get a given app data info 
    Parameters
    ----------------
    datas:              datas from map_data_app function
    result_type:        result type
    
    Return
    ----------------
    output: dictionary containing signal information
    times: signal times
    values: signal values
    """
    output = {}
    
    datas = datas['users'][0]['data']
    times = []
    values = []
    for data in datas:
        if result_type == data['type']:
            times.append(np.datetime64(data['timestamp']))
            values.append(data['values'][0])

    output['times'] = times
    output['values'] = values

    return output


def get_status_message(status):

    status_message = None
    if status == 200:
        status_message = 'Request succeed'
    elif status == 400:
        status_message = 'Error 400: Missing parameter or incorrect value'
    elif status == 401:
        status_message = 'Error 401: Authentiﬁcation nécessaire\
                          (token missing)'
    elif status == 403:
        status_message = 'Error 403: Forbidden (invalid credentials or \
                          no right to access data of end user(s) requested)'
    elif status == 404:
        status_message = 'Error 404: Not available'
    elif status == 500:
        status_message = 'Error 500: Unknown error, contact us'
    elif status == 503:
        status_message = 'Error 503: API is down, and should be available soon'
    else:
        status_message = 'Unknown error'

    return status_message


def time_shift(my_time, offset, time_format='h'):
    my_time = my_time.replace(' ', 'T')
    my_time = np.datetime64(my_time)
    my_time = my_time + np.timedelta64(offset, time_format)
    my_time = str(my_time)
    my_time = my_time.replace('T', ' ')

    return my_time
