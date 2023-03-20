import numpy as np

def get_np_datetime_info(np_datetime):
    ''' Get numpy datetime64 information

    Parameters
    -----------------------
    np_datetime : numpy datetime64

    Returns
    -----------------------
    year, month, day, hour, minute, second (all integers)

    '''

    dt = np.str(np_datetime)
    year = int(dt[:4])
    month = int(dt[5:7])
    day = int(dt[8:10])
    hour = int(dt[11:13])
    minute = int(dt[14:16])
    second = int(dt[17:19])

    return year, month, day, hour, minute, second

def datetime_np2str(np_datetime):
    datetime_str = np.str(np_datetime).replace('T', ' ')
    return datetime_str

def datetime_str2np(datetime_str):
    np_datetime = np.datetime64(datetime_str.replace(' ', 'T'))
    return np_datetime

def get_time_intervals(times):
    """ Get time intervals from times input

    Parameters
    -----------------------
    times : numpy datetime64

    Returns
    -----------------------
    time intervals
    """

    time_intervals = []
    if len(times) > 1:
        diff = (times[1:] - times[:-1])
        to_sec = np.timedelta64(int(1e9), 'ns')
        time_intervals = diff/to_sec

    return time_intervals


def get_time_intervals_unwrap(times):
    """ Get time intervals from times input

    Parameters
    -----------------------
    times : list of numpy datetime64

    Returns
    -----------------------
    list of time intervals
    """

    time_intervals_s = []
    for time in times:
        time_intervals = get_time_intervals(time)
        time_intervals_s.append(time_intervals)

    return time_intervals_s

def get_time_intersection(times_1, times_2):

    if len(times_1) == 0 or len(times_2) == 0:
        return []

    tmin_intersections = []
    tmax_intersections = []

    for t_1 in times_1:
        if len(t_1) == 0:
            continue
        tmin_1 = t_1[0]
        tmax_1 = t_1[-1]
        for t_2 in times_2:
            if len(t_2) == 0:
                continue
            tmin_2 = t_2[0]
            tmax_2 = t_2[-1]

            tmins_up = max(tmin_1, tmin_2)
            tmaxs_low = min(tmax_1, tmax_2)

            if tmins_up < tmaxs_low:
                tmin_intersections.append(tmins_up)
                tmax_intersections.append(tmaxs_low)
    
    output = {}
    output['time_start'] = tmin_intersections
    output['time_stop'] = tmax_intersections
    
    return output

def get_utc_offset(code_zone):
    """ Get time offset according to utc
    Parameters
    ----------------
    country_code

    Return
    ----------------
    time offset according to utc
    """
    utc_offsets = {
        'ACDT': 10.5,
        'ACST': 9.5,
        'ACT':	-5,
        'ACWST': 8.75,
        'ADT':	-3,
        'AEDT': 11,
        'AEST': 10,
        'AFT':	4.5,
        'AKDT': -8,
        'AKST': -9,
        'ALMT': 6,
        'AMST': -3,
        'AMT1':	-4,
        'AMT2': 4,
        'ANAT': 12,
        'AQTT': 5,
        'ART':	-3,
        'AST1':	3,
        'AST2': -4,
        'AWST': 8,
        'AZOST': 0,
        'AZOT': -1,
        'AZT':	4,
        'BDT':	8,
        'BIOT': 6,
        'BIT':	-12,
        'BOT':	-4,
        'BRST': -2,
        'BRT':	-3,
        'BST1':	6,
        'BST2':	11,
        'BST3':	1,
        'BTT':	6,
        'CAT':	2,
        'CCT':	6.5,
        'CDT1':	-5,
        'CDT2':	-4,
        'CEST': 2,
        'CET':	1,
        'CHADT': 13.75,
        'CHAST': 12.75,
        'CHOT': 8,
        'CHOST': 9,
        'CHST': 10,
        'CHUT': 10,
        'CIST': -8,
        'WITA': 8,
        'CKT':	-10,
        'CLST': -3,
        'CLT':	-4,
        'COST': -4,
        'COT':	-5,
        'CST1': -6,
        'CST2': 8,
        'CST3': -5,
        'CVT':	-1,
        'CWST': 8.75,
        'CXT':	7,
        'DAVT': 7,
        'DDUT': 10,
        'DFT':	1,
        'EASST': -5,
        'EAST': -6,
        'EAT':	3,
        'ECT1':	-4,
        'ECT2':	-5,
        'EDT':	-4,
        'EEST': 3,
        'EET':	2,
        'EGST': 0,
        'EGT':	-1,
        'WIT':	9,
        'EST':	-5,
        'FET':	3,
        'FJT':	12,
        'FKST': -3,
        'FKT':	-4,
        'FNT':	-2,
        'GALT': -6,
        'GAMT': -9,
        'GET':	4,
        'GFT':	-3,
        'GILT': 12,
        'GIT':	-9,
        'GMT':	0,
        'GST1':	-2,
        'GST2':	4,
        'GYT':	-4,
        'HDT':	-9,
        'HAEC': 2,
        'HST':	-10,
        'HKT':	8,
        'HMT':	5,
        'HOVST': 8,
        'HOVT': 7,
        'ICT':	7,
        'IDLW': -12,
        'IDT':	3,
        'IOT':	3,
        'IRDT': 4.5,
        'IRKT': 8,
        'IRST': 3.5,
        'IST1':	5.5,
        'IST2':	1,
        'IST3':	2,
        'JST':	9,
        'KALT': 2,
        'KGT':	6,
        'KOST': 11,
        'KRAT': 7,
        'KST':	9,
        'LHST1': 10.5,
        'LHST2': 11,
        'LINT': 14,
        'MAGT': 12,
        'MART': -9.5,
        'MAWT': 5,
        'MDT':	-6,
        'MET':	1,
        'MEST': 2,
        'MHT':	12,
        'MIST': 11,
        'MIT':	-9.5,
        'MMT':	6.5,
        'MSK':	3,
        'MST1':	8,
        'MST2':	-7,
        'MUT':	4,
        'MVT':	5,
        'MYT':	8,
        'NCT':	11,
        'NDT':	-2.5,
        'NFT':	11,
        'NOVT': 7,
        'NPT':	5.75,
        'NST':	-3.5,
        'NT':	-3.5,
        'NUT':	-11,
        'NZDT': 13,
        'NZST': 12,
        'OMST': 6,
        'ORAT': 5,
        'PDT':	-7,
        'PET': -5,
        'PETT': 12,
        'PGT':	10,
        'PHOT': 13,
        'PHT':	8,
        'PKT':	5,
        'PMDT': -2,
        'PMST': -3,
        'PONT': 11,
        'PST1':	-8,
        'PST2':	8,
        'PYST': -3,
        'PYT':	-4,
        'RET':	4,
        'ROTT': -3,
        'SAKT': 11,
        'SAMT': 4,
        'SAST': 2,
        'SBT':	11,
        'SCT':	4,
        'SDT':	-10,
        'SGT':	8,
        'SLST': 5.5,
        'SRET': 11,
        'SRT':	-3,
        'SST1':	-11,
        'SST2':	8,
        'SYOT': 3,
        'TAHT': -10,
        'THA':	7,
        'TFT':	5,
        'TJT':	5,
        'TKT':	13,
        'TLT':	9,
        'TMT':	5,
        'TRT':	3,
        'TOT':	13,
        'TVT':	12,
        'ULAST': 9,
        'ULAT': 8,
        'UTC':	0,
        'UYST': -2,
        'UYT':	-3,
        'UZT':	5,
        'VET': -4,
        'VLAT': 10,
        'VOLT': 4,
        'VOST': 6,
        'VUT':	11,
        'WAKT':	12,
        'WAST': 2,
        'WAT': 1,
        'WEST': 1,
        'WET': 0,
        'WIB': 7,
        'WGST': -2,
        'WGT': -3,
        'WST': 8,
        'YAKT': 9,
        'YEKT':	5,
       }
    
    if code_zone not in utc_offsets.keys():
        raise NameError(code_zone + ' code zone does not exist in utc table')
    offset = utc_offsets[code_zone]
    
    return offset

def second2time(sec):
    h       = sec/3600
    HH      = int(np.floor(h))
    
    m       = (h - HH)*60
    MM      = int(np.floor(m))
    
    s       = (m - MM)*60
    SS      = int(np.floor(s))
    
    if HH < 10:
        prefix = '0'
    else:
        prefix = ''
    HH      = prefix + str(HH)
    
    if MM < 10:
        prefix = '0'
    else:
        prefix = ''
    MM      = prefix + str(MM)
    if SS < 10:
        prefix = '0'
    else:
        prefix = ''
    SS      = prefix + str(SS)
        
    time    = HH + ':' + MM + ':' + SS
    
    return time