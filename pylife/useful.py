import numpy as np
from pylife.env import get_env
DEV = get_env()
# import os
from pylife.time_functions import get_time_intersection
from pylife.QT_Interval import detect_QT_interval
from pylife.detect_ECG_Peaks import getPeaks_unwrap


# --- Add imports for DEV env
# if DEV:
    # import matplotlib.pyplot as plt
    # import smtplib
    # from email.mime.multipart import MIMEMultipart 
    # from email.mime.text import MIMEText 
    # from email.mime.base import MIMEBase 
    # from email import encoders 
    # from reportlab.pdfgen import canvas
    # from reportlab.lib import colors
    # from reportlab.platypus import Table
    # import pandas as pd
    # from openpyxl import load_workbook
    # from zipfile import ZipFile
    # from pptx import Presentation 
    # from pptx.util import Inches, Pt, Cm
    # from pptx.dml.color import RGBColor
    # from pptx.enum.text import PP_ALIGN
  

def get_fs(signal_type):
    """ Get signal sampling frequency from signal type
    Parameters
    ----------------
    signal_type: Signal type

    Return
    ----------------
    Sampling frequency

    """
    fs_ecg = 200
    fs_acc = 50
    fs_breath = 20
    fs_temp = 1
    fs_imp = 1/(10*60)
    sample_frequencies = {'acc': fs_acc, 'accx': fs_acc, 'accy': fs_acc,
                          'accz': fs_acc, 'breath': fs_breath,
                          'breath_1': fs_breath, 'breath_2': fs_breath,
                          'breath_1_filtered': fs_breath, 'breath_2_filtered': fs_breath,
                          'ecg': fs_ecg, 'ecg_filtered': fs_ecg,
                          'temp': fs_temp, 'temp_1': fs_temp, 'temp_2': fs_temp,
                          'temp_valid': fs_temp, 'temp_1_valid': fs_temp, 
                          'temp_2_valid': fs_temp,
                          'imp': fs_imp, 'imp_1': fs_imp,
                          'imp_2': fs_imp, 'imp_3': fs_imp,
                          'imp_4': fs_imp}

    fs = sample_frequencies[signal_type]

    return fs


def get_imp_types():
    """ Returns impendance signal labels """
    return ['imp_1', 'imp_2', 'imp_3', 'imp_4']

def get_signal_types():
    """ Returns signal labels used in database """
    return ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
            'temp_1', 'temp_2', 'temp_1_valid', 'temp_2_valid', 'imp']

def get_signal_filtered_types():
    """ Returns signal filtered labels used in database """
    return ['breath_1_filtered', 'breath_2_filtered', 'ecg_filtered']

def get_signal_result_types():
    """ Returns signal filtered labels used in database """
    return ['heartbeat', 'heartbeat_quality_index',
            'HRV', 'HRV_quality_index',
            'respiratory_rate', 'respiratory_rate_quality_index',
            'activity_level', 'steps_number', 'averaged_activity',
            'averaged_temp_1', 'averaged_temp_2', 'ecg_quality_index',
            'is_worn'
            ]

def get_app_info_types():
    """ Returns get_app_info labels used in database """
    return ['battery', 'ble_disconnected', 'notification']

def get_all_api_types():
    """ Returns all labels used in database """
    return get_signal_types() + get_signal_filtered_types() + get_signal_result_types() + get_app_info_types()

def unwrap(values):
    """ Unwrap values from list

     Parameter
    ----------
    values:     Values to unwrap

    Returns
    ----------
    new_values: unwrapped values

    """
    if not is_list_of_list(values):
        return values

    new_values = []
    for value in values:     
        if value is not None and type(value)==float:            
            new_values.append(value)
        else:
             if value is not None:
                 new_values.extend(value) 

    return np.array(new_values)


def wrap(values, ids):
    """ Wrap values to list from ids

     Parameter
    ----------
    values: Values to wrap
    ids:    reference ids for wrapping

    Returns
    ----------
    list_values: wrapped values

    """
    assert len(values) == len(ids), 'Values, ids should have the same length'

    ids = np.array(ids)
    list_values = []
    index_id_s = []

    imin = 0
    for id_ in np.unique(ids)[1:]:
        id_ = int(id_)
        index_id = np.argwhere(ids == id_)[:, 0][0]
        index_id_s.append(index_id)
        imax = index_id - 1
        list_values.append(values[imin: imax])
        imin = index_id
    list_values.append(values[imin:])

    return list_values


def center(x):
    """ Center signal """
    return x - np.mean(x)


def reduce(x):
    """ Reduce signal """
    y = x
    if np.std(x) != 0:
        y = x / np.std(x)
    return y


def normalize(x, vmin=None, vmax=None):
    """ Normalize signal """
    x = np.array(x)
    
    if vmin is not None and vmax is not None:
        VMIN = min(x)
        VMAX = max(x)
        y = (x - VMIN) * (vmax - vmin) / (VMAX - VMIN) + vmin
        
    else:
        y = x / max(abs(x))
        
    return y


def standardize(x):
    """ Standardize signal """
    y = center(x)
    y = reduce(y)
    # y = normalize(y)
    return y


def convert_sec_to_min_freq(values):
    """ Convert second values to occurence per minute

     Parameter
    ----------
    values:     Values to convert (seconds)

    Return
    ----------
    new_values: Frequencies per minutes

    """
    values = np.array(values)
    values = values[np.isnan(values) == False]
    values_pm = 60/values

    return values_pm


def compute_signal_mean(times, sig):
    """ Compute a weighted average of the signal according
    to the given time intervals.

    :param times: list of numpy arrays, each containing interval
    start and end times
    :param sig: list of numpy arrays containing signal values
    :return: signal weighted average
    """

    signal_sum = 0
    signal_length = 0

    for i, sig_array in enumerate(sig):
        sig_array = np.array(sig_array)
        if not sig_array.size:
            # No data point: does not contribute to the
            # average, we can skip it
            continue

        for j, value in enumerate(sig_array):
            value_duration = times[i][j][1] - times[i][j][0]
            signal_sum += value * value_duration
            signal_length += value_duration

    if signal_length == 0:
        return None

    return signal_sum / signal_length

def is_list_of_list(values):
    """ Define if input signal is a matrix """
    is_list_list = False

    if isinstance(values, int) or  isinstance(0, float):
        return is_list_list

    if len(values) > 0:
        sig = np.array(values)
        for i in range(len(sig)):
            if type(sig[i]) is np.ndarray or type(sig[i]) is list:
                is_list_list = True
                break

    return is_list_list

def get_durations_info(times, sig, fs):
    sig_durations = []
    n_samples_per_sig = []
    first_timestamps = []
    last_timestamps = []
    t = [0]

    is_wrapped = is_list_of_list(sig)
    if is_wrapped:
        for i, seg in enumerate(sig):
            if len(seg) > 0:
                t = t[-1] + np.linspace(0, len(seg)/fs, len(seg))
                sig_durations.append(t[-1] - t[0])
                n_samples_per_sig.append(len(seg))
                first_timestamps.append(times[i][0])
                last_timestamps.append(times[i][-1])
            else:
                sig_durations.append(0)
                n_samples_per_sig.append(len(seg))

    else:
        if len(sig) > 0:
            t = t[-1] + np.linspace(0, len(sig)/fs, len(sig))
            sig_durations.append(t[-1] - t[0])
            n_samples_per_sig.append(len(sig))
            first_timestamps.append(times[0])
        else:
            sig_durations.append(0)
            n_samples_per_sig.append(len(sig))

    sig_durations = np.array(sig_durations)
    n_samples_per_sig = np.array(n_samples_per_sig)
    first_timestamps = np.array(first_timestamps)
    last_timestamps = np.array(last_timestamps)

    if len(first_timestamps) > 1:
        disconnection_durations = (first_timestamps[1:]
                                      - last_timestamps[:-1])
        if type(disconnection_durations[0]) == np.timedelta64:
            disconnection_durations = disconnection_durations\
                / np.timedelta64(1, 's')
        disconnection_durations = np.append(disconnection_durations, 0)
    else:
        disconnection_durations = [0]

    output = {}
    output['sig_durations'] = sig_durations
    output['n_samples_per_sig'] = n_samples_per_sig
    output['first_timestamps'] = first_timestamps
    output['last_timestamps'] = last_timestamps
    output['disconnection_durations'] = disconnection_durations

    return output

def get_sig_loss(disconnection_durations, sig_durations):

    t_out = np.array(sum(disconnection_durations))
    if t_out.ndim > 0:
        t_out = t_out[0]
    t_in = np.array(sum(sig_durations))
    if t_in.ndim > 0:
        t_in = t_in[0]

    loss = t_out/(t_in + t_out)*100

    return loss

def get_stats_clean_sig(indicators, fs):
    """ Set stats for clean signal:
        sig_clean_n_sample_per_segment : number of cleaned sample
        sig_clean_percentage_per_segment : percentage of cleaned sample
    """

    output = {}
    output['n_sample'] = None
    output['n_segments'] = None
    output['percentage'] = None
    output['duration_min'] = None
    output['duration_max'] = None
    output['duration_median'] = None
    output['duration_iqr'] = None
    output['duration_mean'] = None
    output['duration_std'] = None
    output['duration'] = None
    
    if len(indicators) > 0:
        if is_list_of_list(indicators):
            indicators = unwrap(indicators)

        count = 0
        counts = []
        for ind in indicators:
            if ind == 1:
                count += 1
            else:
                if count > 0:
                    counts.append(count)
                    count = 0
        if count > 0:
            counts.append(count)
        if len(counts) == 0:
            counts = [count]
        counts = np.array(counts)
        n_clean = sum(counts)
        duration_min = min(counts)/fs
        duration_max = max(counts)/fs
        duration_median = np.median(counts)/fs
        duration_iqr = (np.percentile(counts, 75) - np.percentile(counts, 25))/fs
        duration_mean = np.mean(counts)/fs
        duration_std = np.std(counts)/fs
        percentage = n_clean / len(indicators) * 100
    
        output['n_sample'] = n_clean
        output['percentage'] = percentage
        output['duration_min'] = duration_min
        output['duration_max'] = duration_max
        output['duration_median'] = duration_median
        output['duration_iqr'] = duration_iqr
        output['duration_mean'] = duration_mean
        output['duration_std'] = duration_std
        output['duration'] = n_clean / fs

    return output

def get_stats_not_clean_sig(indicators, fs):
    """ Set stats for clean signal:
        sig_clean_n_sample_per_segment : number of cleaned sample
        sig_clean_percentage_per_segment : percentage of cleaned sample
    """

    output = {}
    output['n_sample'] = None
    output['n_segments'] = None
    output['percentage'] = None
    output['duration_min'] = None
    output['duration_max'] = None
    output['duration_median'] = None
    output['duration_iqr'] = None
    output['duration_mean'] = None
    output['duration_std'] = None
    output['duration'] = None
    
    if len(indicators) > 0:
        if is_list_of_list(indicators):
            indicators = unwrap(indicators)

        count = 0
        counts = []
        for ind in indicators:
            if ind == 0:
                count += 1
            else:
                if count > 0:
                    counts.append(count)
                    count = 0
        if count > 0:
            counts.append(count)
        if len(counts) == 0:
            counts = [count]
        counts = np.array(counts)
        n_not_clean = sum(counts)
        duration_min = min(counts)/fs
        duration_max = max(counts)/fs
        duration_median = np.median(counts)/fs
        duration_iqr = (np.percentile(counts, 75) - np.percentile(counts, 25))/fs
        duration_mean = np.mean(counts)/fs
        duration_std = np.std(counts)/fs
        percentage = n_not_clean / len(indicators) * 100
    
        output['n_sample'] = n_not_clean
        output['percentage'] = percentage
        output['duration_min'] = duration_min
        output['duration_max'] = duration_max
        output['duration_median'] = duration_median
        output['duration_iqr'] = duration_iqr
        output['duration_mean'] = duration_mean
        output['duration_std'] = duration_std
        output['duration'] = n_not_clean / fs

    return output

def get_stats_clean_sig_intersection(times_clean1, times_clean2, times_ref, fs_ref):
    
    times_ref_unwrap = np.array(unwrap(times_ref))
    indicators = np.zeros(len(times_ref_unwrap))
    times_intersections = []    
    if len(times_clean1) != 0 and len(times_clean2) != 0:
        time_inter = get_time_intersection(times_clean1, times_clean2)
        if len(time_inter['time_start']) != 0:
            for i in range(len(time_inter['time_start'])):
                t0 = time_inter['time_start'][i]
                t1 = time_inter['time_stop'][i]
                times_intersections.append(np.arange(t0, t1, np.timedelta64(int(1e6/fs_ref), 'us')))
                
            for i in range(len(times_intersections)):
                imin = np.where(times_ref_unwrap >= (times_intersections[i][0]))[0][0]
                imax = np.where(times_ref_unwrap >= (times_intersections[i][-1]))[0][0]
                indicators[imin:imax] = 1
    
    output = get_stats_clean_sig(indicators, fs_ref)
    
    indicators_wrap = []
    imin = 0
    for i in range(len(times_ref)):
        imax = imin + len(times_ref[i])
        indicators_wrap.append(indicators[imin:imax])
        imin = imax
        
    output['times'] = times_intersections
    output['indicators'] = indicators_wrap

    return output

def get_stats_not_clean_sig_intersection(times_clean1, times_clean2, times_ref, fs_ref):
    
    times_ref_unwrap = np.array(unwrap(times_ref))
    indicators = np.zeros(len(times_ref_unwrap))
    times_intersections = []    
    if len(times_clean1) != 0 and len(times_clean2) != 0:
        time_inter = get_time_intersection(times_clean1, times_clean2)
        if len(time_inter['time_start']) != 0:
            for i in range(len(time_inter['time_start'])):
                t0 = time_inter['time_start'][i]
                t1 = time_inter['time_stop'][i]
                times_intersections.append(np.arange(t0, t1, np.timedelta64(int(1e6/fs_ref), 'us')))
                
            for i in range(len(times_intersections)):
                imin = np.where(times_ref_unwrap >= (times_intersections[i][0]))[0][0]
                imax = np.where(times_ref_unwrap >= (times_intersections[i][-1]))[0][0]
                indicators[imin:imax] = 1
    
    output = get_stats_not_clean_sig(indicators, fs_ref)
    
    indicators_wrap = []
    imin = 0
    for i in range(len(times_ref)):
        imax = imin + len(times_ref[i])
        indicators_wrap.append(indicators[imin:imax])
        imin = imax
        
    output['times'] = times_intersections
    output['indicators'] = indicators_wrap

    return output

def get_peaks_from_peaks_times(times, peaks_times):

    peaks = []
    if len(peaks_times) == 0:
        return peaks

    if is_list_of_list(times):
        times = unwrap(times)

    if is_list_of_list(peaks_times):
        peaks_times = unwrap(peaks_times)

    for ip in range(len(peaks_times)):
        peak = np.where(times == peaks_times[ip])[0]
        if len(peak) > 0:
            peaks.append(peak[0])

    return peaks

def set_list_of_list(x):

    if len(x) > 0:
        if not is_list_of_list(x):
            x = [x]
    return np.array(x)

def convert_mv(sig, gain):
    # Gain of system
    gain_analog = 1/4 
    # Reference voltage of system
    ref_volt = .6*1e3 # mV
    # System voltage
    system_volt = ref_volt/gain_analog
    # Resolution
    resolution = 2**12 - 1
    
    sig = np.array(sig)
    sig = sig - np.mean(sig)
    
    # Digital to analog conversion
    sig_mv = sig * system_volt / (resolution * gain)
    
    return sig_mv 

def convert_mv_unwrap(sig, gain):
    sig_mv = []
    for s in sig:
        sig_mv.append(convert_mv(s, gain))
    
    return sig_mv 


# def send_email(to, end_user, from_time=None, to_time=None, filename=None, activity_type=None, project=None):
#     sender = 'clife.data@gmail.com'
#     password = 'chronolifewifi'
#     new_line = '\n'
#     for receiver in to:
#         # instance of MIMEMultipart 
#         msg = MIMEMultipart() 
          
#         # storing the senders email address   
#         msg['From'] = sender 
          
#         # storing the receivers email address
#         msg['To'] = receiver
          
#         # storing the subject  
#         msg['Subject'] = "Data available for user: " + end_user
          
#         # string to store the body of the mail 
#         body = "Data is available in database for user " + end_user
            
#         if filename is not None: 
#             # string to store the body of the mail 
#             body = "Please find the data analysis report for user " + end_user + new_line
             
#             # open the file to be sent  
#             attachment = open(filename, "rb") 
              
#             # instance of MIMEBase and named as p 
#             p = MIMEBase('application', 'octet-stream') 
              
#             # To change the payload into encoded form 
#             p.set_payload((attachment).read()) 
              
#             # encode into base64 
#             encoders.encode_base64(p) 
               
#             ext = '.' + filename.split('.')[-1]
#             p.add_header('Content-Disposition', "attachment; filename= %s" % (end_user + ext)) 
          
#             # attach the instance 'p' to instance 'msg' 
#             msg.attach(p) 
        
#         if project is not None:
#             body += new_line + 'Project: ' + project + new_line
#         if activity_type is not None:
#             body += new_line + 'Activity: ' + activity_type + new_line
#         if from_time is not None and to_time is not None:
#             body += new_line + 'Time request: from ' + from_time + ' to ' + to_time + new_line
            
#         # attach the body with the msg instance 
#         msg.attach(MIMEText(body, 'plain')) 
        
#         # Converts the Multipart msg into a string 
#         text = msg.as_string() 
        
#         try:
#             smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
#         except Exception as e:
#             print(e)
#             smtpObj = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        
#         smtpObj.ehlo()
#         smtpObj.starttls()
#         smtpObj.login(sender, password) 
#         smtpObj.sendmail(sender, receiver, text) # Or recipient@outlook
#         smtpObj.quit()
    
#     print('------------------------------------------------------------------')
#     print('Email sent to', to)

def find_max_length_clean(bin_ind, fs):
    
    max_clean = 0
    length_1 = 0
    for j in range(len(bin_ind)):
        a = bin_ind[j]
        i = 0
        while i < len(a):
            length_1 = 0
            while a[i]==1 and i+1 < len(a):
                length_1+=1
                i+=1
                if max_clean < length_1:
                    max_clean = length_1
            i+=1
    return max_clean/fs

def get_median_length_clean(bin_ind, fs):
    length_1 = 0
    ls = []
    for j in range(len(bin_ind)):
        a = bin_ind[j]
        i = 0
        while i < len(a):
            length_1 = 0
            while a[i]==1 and i+1 < len(a):
                length_1+=1
                i+=1
            if a[i]==1 and i+1 == len(a):
                length_1+=1
            ls.append(length_1)
            i+=1
    ls = np.array(ls)
    ls = ls[ls >0]
    return np.median(ls)/fs


# def pdf_report(report, path_save, operator=None, end_user=None, 
#                from_time=None, to_time=None, 
#                window=None, activity_type=None, project=None, 
#                flag_clean=None, flag_time_split=True, 
#                flag_disconnection_detail=False,
#                flag_random_segments=True):

#     # file_report = report['filename']
#     # path_data = os.path.dirname(file_report) + '/'
#     filename = path_save + end_user
#     if activity_type is not None:
#         filename += '_' + activity_type
#     filename +=  '_report.pdf'
#     doc_title = 'Doc title'
#     title = 'Chronolife Data Report'
#     pdf = canvas.Canvas(filename)
        
#     # % Document title
#     pdf.setTitle(doc_title)
    
#     h1          = 26
#     h2          = 16
#     h3          = 12
#     xh1         = 290
#     yh1         = 790
#     yjumpline   = 15
#     xpos_left   = 10
#     xpos_right  = 300
#     xpos_center = int(.5*(xpos_left + xpos_right))
#     xpos_xl     = 80
    
#     height      = 240
#     width       = 300
#     height_xl   = int(height*1.5)
#     width_xl    = int(width*1.5)
    
#     # Title
#     pdf.setFont("Courier-Bold", h1)
#     ypos = yh1
#     pdf.drawCentredString(xh1, ypos, title)
    
#     # General information
#     pdf.setFont("Courier-Bold", h3)
#     ypos -= 50
#     if from_time is not None:
#         pdf.drawCentredString(xh1, ypos, 'General informaiton')
#     pdf.setFont("Courier", h3)
#     # Project
#     if project is not None:
#         ypos -= 15
#         pdf.drawCentredString(xh1, ypos, 'Project: ' + project)
#     # Activity type
#     if activity_type is not None:
#         ypos -= 15
#         pdf.drawCentredString(xh1, ypos, 'Activity: ' + activity_type)
#     # user
#     if end_user is not None:
#         ypos -= 15
#         pdf.drawCentredString(xh1, ypos, 'End-user: ' + end_user)
    
    
#     # Infos
#     pdf.setFont("Courier-Bold", h3)
#     ypos -= 30
#     if from_time is not None:
#         pdf.drawCentredString(xh1, ypos, 'Report informaiton')
#     pdf.setFont("Courier", h3)
#     # Operator
#     if operator is not None:
#         pdf.setFont("Courier", h3)
#         ypos -= 15
#         pdf.drawCentredString(xh1,  ypos, 'Operator: ' + operator)
#     # Date
#     pdf.setFont("Courier", h3)
#     ypos -= 15
#     now = str(np.datetime64('now', 'm')).replace('T', ' ')
#     pdf.drawCentredString(xh1,      ypos, 'Date: ' + now)
    
#     # Request time
#     pdf.setFont("Courier-Bold", h3)
#     ypos -= 30
#     if from_time is not None:
#         pdf.drawCentredString(xh1, ypos, 'Date requested in database')
#     pdf.setFont("Courier", h3)
#     ypos -= 15
#     if from_time is not None:
#         pdf.drawCentredString(xh1, ypos, 'From  ' + from_time)
#     ypos -= 15
#     if to_time is not None:
#         pdf.drawCentredString(xh1, ypos, 'To    ' + to_time)                              
    
#     # Summary 
#     pdf.setFont("Courier-Bold", h2)
#     ypos -= 50
#     pdf.drawCentredString(xh1, ypos, 'Summary')
#     pdf.setFont("Courier", h3)
#     ypos -= yjumpline 
#     pdf.drawCentredString(xh1, ypos, 'Durations are given in seconds')
        
#     style = [('ALIGN', (0,0), (-1,-1), 'CENTER'),
#           ('LINEBELOW', (0,0), (-1,0), 1, colors.darkcyan),
#           ('FONT', (0,0), (-1,-1), 'Times-Roman', 12),
#           ('FONT', (0,0), (-1,0), 'Times-Bold'),
#           ('FONT', (0,0), (0,-1), 'Times-Bold'),
#           ]
    
#     df_ini = report['summary']
#     keep_cols = ['start', 'stop', 'duration']
#     df = df_ini[keep_cols]
#     df = df[df[df.columns[0]].isnull() == False]
#     df.duration = df.duration.astype('float').round().astype('int32')
#     cols = df.columns.values.astype(str).tolist()
#     cols = np.insert(cols, 0, '')
#     index = np.matrix(df.index)
#     values = df.values
#     rows = np.hstack((index.T, values)).tolist()
#     lista = [cols] + rows
    
#     table = Table(lista, style=style)
#     table.wrapOn(pdf, 100, 100)
#     ypos -= 200
#     table.drawOn(pdf, 140, ypos)
    
#     ypos -= height + 20
#     if flag_clean:
#         pdf.drawInlineImage(path_save + 'global_disconnection_bar.png', x=xpos_left, y=ypos, width=width, height=height)
#         pdf.drawInlineImage(path_save + 'global_usable_bar.png', x=xpos_right, y=ypos, width=width, height=height)
#     else:
#         pdf.drawInlineImage(path_save + 'global_disconnection_bar.png', x=xpos_center, y=ypos, width=width, height=height)
    
#     if flag_disconnection_detail:
#         # Disconnection detail
#         pdf.showPage()
#         pdf.setFont("Courier-Bold", h2)
#         ypos = yh1
#         pdf.drawCentredString(xh1, ypos, 'Disconnection detail')
#         pdf.setFont("Courier", h3)
#         ypos -= yjumpline 
#         pdf.drawCentredString(xh1, ypos, 'Durations are given in seconds')
        
#         df_ini = report['disconnection_detail']
#         keep_cols = ['start', 'stop', 'duration']
#         df = df_ini[keep_cols]
#         df = df[df[df.columns[0]].isnull() == False]
#         if len(df) > 0:
#             count = 0
#             idx_sig_types = np.sort(df.index.value_counts().index)
#             for idx in idx_sig_types:
#                 count += 1
#                 if count > 1:
#                     pdf.showPage()
#                     pdf.setFont("Courier-Bold", h2)
#                     ypos = yh1
#                     pdf.drawCentredString(xh1, ypos, 'Disconnection detail')
#                     pdf.setFont("Courier", h3)
#                     ypos -= yjumpline 
                
#                 pdf.drawCentredString(xh1, ypos, 'Durations are given in seconds')
#                 pdf.setFont("Courier-Bold", h3)
#                 ypos -= 30
#                 pdf.drawCentredString(xh1, ypos, idx)
                    
#                 df_sig = df.loc[idx]
#                 if type(df_sig) == pd.Series:
#                     df_sig = df_sig.to_frame().T
#                 df_sig.duration = df_sig.duration.astype('float').round().astype('int32')
#                 cols = df.columns.values.astype(str).tolist()
#                 cols = np.insert(cols, 0, '')
                
#                 index = np.matrix(df_sig.index)
#                 values = df_sig.values
#                 rows = np.hstack((index.T, values)).tolist()
#                 lista = [cols] + rows
                
#                 table = Table(lista, style=style)
#                 table.wrapOn(pdf, 100, 100)
#                 ypos -= (150 + len(df_sig)*17)
#                 table.drawOn(pdf, 140, ypos)
#         else:
#             ypos -= 50
#             pdf.drawCentredString(xh1, ypos, 'No disconnections.')
    
    # # Usable signal table
    # pdf.showPage()
    # pdf.setFont("Courier-Bold", h2)
    # ypos = yh1
    # pdf.drawCentredString(xh1, ypos, 'Usable signal')
    
    # # df_ini = pd.read_excel(file_report, sheet_name='usable')
    # df_ini = report['usable']
    # keep_cols = ['percentage', 'duration', 'duration_median', 'duration_iqr']
    # df = df_ini[keep_cols]
    # cols = df.columns.values.astype(str).tolist()
    # cols = np.insert(cols, 0, '')
    # index = np.matrix(df_ini.index)
    
    # values = df.values
    # rows = np.hstack((index.T, values)).tolist()
    # lista = [cols] + rows
    
    # table = Table(lista, style=style)
    # table.wrapOn(pdf, 100, 100)
    # ypos -= 250
    # table.drawOn(pdf, 60, ypos)
    
    # ypos -= height_xl + 20
    # pdf.drawInlineImage(path_save + 'global_usable_bar.png', x=xpos_xl, y=ypos, width=width_xl, height=height_xl)
    
    # artifact table
    # pdf.showPage()
    # pdf.setFont("Courier-Bold", h2)
    # ypos = yh1
    # ypos -= 50
    # pdf.drawCentredString(xh1, ypos, 'artifact')
    
    # # df_ini = pd.read_excel(file_report, sheet_name='artifact')
    # df_ini = report['artifact']
    # keep_cols = ['percentage', 'duration', 'duration_median', 'duration_iqr']
    # df = df_ini[keep_cols]
    # cols = df.columns.values.astype(str).tolist()
    # cols = np.insert(cols, 0, '')
    # index = np.matrix(df_ini.index)
    # values = df.values
    # rows = np.hstack((index.T, values)).tolist()
    # lista = [cols] + rows
    
    # table = Table(lista, style=style)
    # table.wrapOn(pdf, 100, 100)
    # ypos -= 250
    # table.drawOn(pdf, 60, ypos)
    
    # ypos -= height_xl + 20
    # pdf.drawInlineImage(path_save + 'global_artifact_bar.png', x=xpos_xl, y=ypos, width=width_xl, height=height_xl)
    
    # if flag_time_split:
    #     # Usable signal percentage barplot
    #     pdf.showPage()
    #     pdf.setFont("Courier-Bold", h2)
    #     ypos = yh1
    #     pdf.drawCentredString(xh1, ypos, 'Usable signal percentage barplot')
    #     ypos -= yjumpline 
    #     pdf.drawCentredString(xh1, ypos, 'every ' + str(window) + ' minutes')
    #     ypos -= height + 20
    #     pdf.drawInlineImage(path_save + 'usable_acc_bar.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'usable_ecg_bar.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'usable_breath_1_bar.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'usable_breath_2_bar.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'usable_temp_1_bar.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'usable_temp_2_bar.png', x=xpos_right, y=ypos, width=width, height=height)

    #     # BPM, RPM, Step number and Activity values
    #     pdf.showPage()
    #     pdf.setFont("Courier-Bold", h2)
    #     ypos = yh1
    #     pdf.drawCentredString(xh1, ypos, 'BPM, RPM, Step number and Activity values')
    #     ypos -= yjumpline 
    #     pdf.drawCentredString(xh1, ypos, 'every ' + str(window) + ' minutes')
    #     ypos -= height + 20
    #     pdf.drawInlineImage(path_save + 'result_bpm_bar.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'result_hrv_bar.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'result_rpm_tho_bar.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'result_rpm_abd_bar.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'result_n_steps_bar.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'result_mean_activity_bar.png', x=xpos_right, y=ypos, width=width, height=height)
        
    #     # Usable signal percentage distribution
    #     pdf.showPage()
    #     pdf.setFont("Courier-Bold", h2)
    #     ypos = yh1
    #     pdf.drawCentredString(xh1, ypos, 'Usable signal percentage distribution')
    #     ypos -= yjumpline 
    #     pdf.drawCentredString(xh1, ypos, 'every ' + str(window) + ' minutes')
    #     ypos -= height + 20
    #     pdf.drawInlineImage(path_save + 'usable_acc_dist.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'usable_ecg_dist.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'usable_breath_1_dist.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'usable_breath_2_dist.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'usable_temp_1_dist.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'usable_temp_2_dist.png', x=xpos_right, y=ypos, width=width, height=height)
        
    #     # BPM, RPM, Step number and Activity distributions
    #     pdf.showPage()
    #     pdf.setFont("Courier-Bold", h2)
    #     ypos = yh1
    #     pdf.drawCentredString(xh1, ypos, 'BPM, RPM, Step number and Activity distributions')
    #     ypos -= yjumpline 
    #     pdf.drawCentredString(xh1, ypos, 'every ' + str(window) + ' minutes')
    #     ypos -= height + 20
    #     pdf.drawInlineImage(path_save + 'result_bpm_dist.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'result_hrv_dist.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'result_rpm_tho_dist.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'result_rpm_abd_dist.png', x=xpos_right, y=ypos, width=width, height=height)
    #     ypos -= height + 10
    #     pdf.drawInlineImage(path_save + 'result_n_steps_dist.png', x=xpos_left, y=ypos, width=width, height=height)
    #     pdf.drawInlineImage(path_save + 'result_mean_activity_dist.png', x=xpos_right, y=ypos, width=width, height=height)
        
    #     # ECG usable signal random segments
    #     if flag_random_segments:
    #         pdf.showPage()
    #         pdf.setFont("Courier-Bold", h2)    
    #         ypos = yh1
    #         pdf.drawCentredString(xh1, ypos, 'ECG usable signal random segments')
    #         ypos -= height + 20
    #         pdf.drawInlineImage(path_save + 'ecg_random_seg0.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'ecg_random_seg1.png', x=xpos_right, y=ypos, width=width, height=height)
    #         ypos -= height + 10
    #         pdf.drawInlineImage(path_save + 'ecg_random_seg2.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'ecg_random_seg3.png', x=xpos_right, y=ypos, width=width, height=height)
    #         ypos -= height + 10
    #         pdf.drawInlineImage(path_save + 'ecg_random_seg4.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'ecg_random_seg5.png', x=xpos_right, y=ypos, width=width, height=height)
            
    #         # Thoracic breathing usable signal random segments
    #         pdf.showPage()
    #         pdf.setFont("Courier-Bold", h2)
    #         ypos = yh1
    #         pdf.drawCentredString(xh1, ypos, 'Thoracic breathing usable signal random segments')
    #         ypos -= height + 20
    #         pdf.drawInlineImage(path_save + 'breath_1_random_seg0.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'breath_1_random_seg1.png', x=xpos_right, y=ypos, width=width, height=height)
    #         ypos -= height + 10
    #         pdf.drawInlineImage(path_save + 'breath_1_random_seg2.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'breath_1_random_seg3.png', x=xpos_right, y=ypos, width=width, height=height)
    #         ypos -= height + 10
    #         pdf.drawInlineImage(path_save + 'breath_1_random_seg4.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'breath_1_random_seg5.png', x=xpos_right, y=ypos, width=width, height=height)
            
    #         # Abdominal breathing usable signal random segments
    #         pdf.showPage()
    #         pdf.setFont("Courier-Bold", h2)
    #         ypos = yh1
    #         pdf.drawCentredString(xh1, ypos, 'Abdominal breathing usable signal random segments')
    #         ypos -= height + 20
    #         pdf.drawInlineImage(path_save + 'breath_2_random_seg0.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'breath_2_random_seg1.png', x=xpos_right, y=ypos, width=width, height=height)
    #         ypos -= height + 10
    #         pdf.drawInlineImage(path_save + 'breath_2_random_seg2.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'breath_2_random_seg3.png', x=xpos_right, y=ypos, width=width, height=height)
    #         ypos -= height + 10
    #         pdf.drawInlineImage(path_save + 'breath_2_random_seg4.png', x=xpos_left, y=ypos, width=width, height=height)
    #         pdf.drawInlineImage(path_save + 'breath_2_random_seg5.png', x=xpos_right, y=ypos, width=width, height=height)
    
    # pdf.save()
    
    # return filename

# def xls_concat_reports_by_activity(path_user, user, filename, sheet_names, on=None, activity_thr=None, activity_types=None, path_save=None):
    
#     keys = sheet_names
#     foldlist = os.listdir(path_user)
#     reports = {}
#     for key in keys:
#         reports[key] = pd.DataFrame()

#     for fold in foldlist:
#         path_reports = path_user + fold + '/'  
#         if not os.path.isdir(path_reports):
#             continue
#         for key in keys:
#             if activity_types is not None:
#                 for activity_type in np.unique(activity_types):
#                     if activity_type.lower() in path_reports.split('/')[-2].lower():
#                         report = pd.read_excel(path_reports + user + '_' + filename + '.xlsx', sheet_name=key)
#                         if on=='rest':
#                             report = report[report.sma_values < activity_thr]
#                         elif on == 'activity':
#                             report = report[report.sma_values >= activity_thr]
#                         report.index = report[report.columns[0]]
#                         report = report.drop([report.columns[0]], axis=1)
#                         report['activity'] = activity_type
#                         report['user'] = user
#                         reports[key] = reports[key].append(report)
            
    
#     if path_save is not None:
        
#         if on is not None:
#             reports['filename'] = path_save + 'concat_' + filename + '_' + on + '.xlsx' 
#         else:
#             reports['filename'] = path_save + 'concat_' + filename + '.xlsx' 
#         with pd.ExcelWriter(reports['filename']) as writer:
#             for key in keys:
#                 reports[key].to_excel(writer, sheet_name=key)
            
#     return reports

# def xls_concat_reports_by_user(path_user, users, activity, path_save=None):
    
#     keys = ['summary', 'disconnection', 'usable', 'result']
    
#     reports = {}
#     for key in keys:
#         reports[key] = pd.DataFrame()
#     filename = 'Report.xlsx'
    
#     for user in users:
#         path_reports = path_user + user + '/sessions/' + activity + '/'  
#         if not os.path.isdir(path_reports):
#             continue
#         for key in keys:
#             report = pd.read_excel(path_reports + filename, sheet_name=key)
#             if len(report.index) > 0:
#                 report.index = report[report.columns[0]]
#                 report = report.drop([report.columns[0]], axis=1)
#                 report['user'] = user
#                 reports[key] = reports[key].append(report)
                

#     if path_save is not None:
#         reports['filename'] = path_save + filename
#         with pd.ExcelWriter(reports['filename']) as writer:
#             for key in keys:
#                 reports[key].to_excel(writer, sheet_name=key)
#                 reports['activities'] = pd.Series(activity, name='Activities')
#                 reports['activities'].to_excel(writer, sheet_name='activities')
            
#     return reports

# def xls_mean_report(reports, sheet_names, activity_types=None, path_save=None):
    
#     filename = reports['filename']
#     keys = sheet_names
#     report_mean = {}
    
#     for key in keys:
#         report_mean[key] = pd.DataFrame(columns=reports[key].columns, index=reports[key].index.unique())
#         for idx in reports[key].index.unique():
#             df_idx = reports[key].loc[idx]
#             if type(df_idx) == pd.Series:
#                 df_idx = pd.DataFrame(np.matrix(df_idx.values), columns=df_idx.index, index=[df_idx.name])
#             for col in df_idx.columns[4:]:
#                 try:
#                     report_mean[key][col].loc[idx] = df_idx[col][df_idx[col].isnull() == False].astype('float').mean()
#                 except:
#                     report_mean[key][col].loc[idx] = None
#             for col in df_idx.columns[:4]:
#                 report_mean[key][col].loc[idx] = df_idx[col].values[0]
    
#     if path_save is not None:
#         book = load_workbook(filename)
#         with pd.ExcelWriter(filename, engine='openpyxl') as writer:
#             writer.book = book
#             for key in keys:
#                 report_mean[key].to_excel(writer, sheet_name=key + '_mean')
#             writer.save()
    


# def xls_mean_report2(report, activity_types=None, path_save=None):
    
#     filename= report['filename']
#     usable_mean = None
#     result_mean = None
#     columns = ['label', 'mean', 'std']
#     usable = report['usable'] 
#     labels = usable.label.unique()
#     usable_mean = pd.DataFrame(columns=columns)
#     for label in labels:
#         lab = usable[usable.label == label]
#         data = np.matrix([label, lab.percentage.mean().round(1), lab.percentage.std().round(1)])
#         usable_mean = usable_mean.append(pd.DataFrame(data, columns=columns))
    
#     result = report['result'] 
#     labels = result.label.unique()
#     result_mean = pd.DataFrame(columns=columns)
#     for label in labels:
#         lab = result[result.label == label]
#         data = np.matrix([label, lab['value'].mean(), lab['value'].std()])
#         result_mean = result_mean.append(pd.DataFrame(data, columns=columns))
        
#     result = report['result'] 
#     labels = result.label.unique()
#     result_sum = pd.DataFrame(columns=['label', 'sum'])
#     label = 'steps'
#     lab = result[result.label == label]
#     data = np.matrix([label, lab['value'].sum()])
#     result_sum = result_sum.append(pd.DataFrame(data, columns=['label', 'sum']))
        
#     if path_save is not None:
#         book = load_workbook(filename)
#         with pd.ExcelWriter(filename, engine='openpyxl') as writer:
#             writer.book = book
#             usable_mean.to_excel(writer, sheet_name='usable_mean')
#             result_mean.to_excel(writer, sheet_name='result_mean')
#             result_sum.to_excel(writer, sheet_name='result_sum')
#             writer.save()
        
# def zip_reports(path_user, user):
#     filename_save = path_user + user + '.zip'
#     with ZipFile(filename_save, 'w') as zipObj:
        
#         for filename in os.listdir(path_user):
#             if 'pdf' not in filename: # and 'xls' not in filename:
#                 continue
#             filePath = os.path.join(path_user, filename)
#             zipObj.write(filePath, filename)
                
#         foldlist = os.listdir(path_user)
#         for fold in foldlist:
#             path_reports = path_user + fold + '/'  
#             if not os.path.isdir(path_reports):
#                 continue
            
#             for filename in os.listdir(path_reports):
#                 if 'pdf' not in filename: # and 'xls' not in filename:
#                     continue
#                 filePath = os.path.join(path_reports, filename)
#                 zipObj.write(filePath, os.path.join(fold, filename))
#     return filename_save


# def transform_indicators_seconds(times, indicators, fs):
#     # return a list where for each second the value is calculated as below :
#     # if there are values on more than 80% of the signal in one second then 1, else 0
#     # it also return a list of date time tags corresponding to the starts of the seconds in question
    
#     timeline    = np.array(unwrap(times))
#     indicators  = np.array(indicators)
#     t = timeline[0]
#     new_t = []
#     new_indicators = []
#     while t < timeline[-1]:
#         t_delta = timeline[timeline > t]
#         ind_delta = indicators[timeline > t]

#         ind_delta = ind_delta[t_delta < t + np.timedelta64(1, 's')]
#         t_delta = t_delta[t_delta < t + np.timedelta64(1, 's')]
#         new_t.append(t)
#         if np.sum(ind_delta) > 80*fs/100:
#              new_indicators.append(1)
#         else:
#              new_indicators.append(0) 
#         t = t + np.timedelta64(1, 's')
#     return new_t, new_indicators

def method_window_np(sig, window_s, fs=200, method = 'max'):
    """
    apply methods on consecutive windows and return an array of same size as sig with the results
    
    """
    window           = int(fs*window_s)
    
    nb_windows       =  int(len(sig)/window)
    sigtemp  = np.array(sig)
    temp_val = np.array(sig)
   
   
    if method == 'min':
        #for all segments of one minute get median over minute 
        for i in range( nb_windows ):
            temp_val[i * window :(i+1) * window ] =  min(sigtemp[i * window : (i+1) * window])
        #fill the end that lasts less than a minute 
        if (len(sig)> nb_windows * window):
            endmed = min( sigtemp[(nb_windows * window +1) :])
            temp_val[nb_windows * window:len(sig)-1] = endmed
            
    if method == 'max':
        #for all segments of one minute get median over minute 
        for i in range( nb_windows ):
            temp_val[i * window :(i+1) * window ] =  max(sigtemp[i * window : (i+1) * window])
        #fill the end that lasts less than a minute 
        if (len(sig)> nb_windows * window):
            endmed = max( sigtemp[(nb_windows * window +1) :])
            temp_val[nb_windows * window:len(sig)-1] = endmed
            
    if method == 'sum':
         #for all segments of one minute get median over minute 
         for i in range( nb_windows ):
             temp_val[i * window :(i+1) * window ] =  sum(sigtemp[i * window : (i+1) * window])
         #fill the end that lasts less than a minute 
         if (len(sig)> nb_windows * window):
             endmed = sum( sigtemp[(nb_windows * window +1) :])
             temp_val[nb_windows * window:len(sig)-1] = endmed
            
    return temp_val
 
def transform_indicators_seconds(times, indicators, fs):
    # return a list where for each second the value is calculated as below :
    # if there are values on more than 80% of the signal in one second then 1, else 0
    # it also return a list of date time tags corresponding to the starts of the seconds in question
    
    #test function
    # a = transform_indicators_seconds(times= [[x for x in range(12)]], indicators =[[1,0,1,1,1, 1,0,0,0,0,0, 0]] , fs=3)
    # print(a)
    #### changed for much faster computation
    new_t = []
    new_indicators = []
    # print('len(times)', len(times))
    # print('len(times[0])', len(times[0]))
    # print('len(indicators)', len(indicators))
    # print('len(indicators[0])', len(indicators[0]))
    for i in range (len(times)):
        timeline    = np.array(times[i])
        indicators_i  = np.array(indicators[i])
        #print('indicators_i', indicators_i)
        indicatorsum = method_window_np(indicators_i, window_s=1, fs=fs, method = 'sum')
        #print(indicatorsum)
        indicatorsum_bin = indicatorsum<0.8*fs
        #print('indicatorsum_bin', indicatorsum_bin)
        
        new_t.append([timeline[i*fs] for i in range(int(len(timeline)/fs))])
        new_indicators.append([int(indicatorsum_bin[i*fs]) for i in range(int(len(indicatorsum_bin)/fs))])
        
       
    return unwrap(new_t), unwrap(new_indicators)


def count_usable_segments(indicators, fs, window_smooth_indicators, duration):
    
    indicators = np.array(unwrap(indicators))
    ### Creation des nouveaux indicateurs
    new_indicators = indicators
    
    sep = np.where((indicators[1:]-indicators[:-1]) != 0)[0] + 1
    segs = np.split(indicators, sep)
    
    new_indicators = []
    for seg in segs:
         if len(seg) < window_smooth_indicators*fs: 
              seg = np.ones((len(seg), ))
         new_indicators.append(seg)
    new_indicators = np.array(unwrap(new_indicators))
    
    ## Separation des segments en clean et noisy
    new_sep = np.where((new_indicators[1:]-new_indicators[:-1]) != 0)[0] + 1
    segs = np.split(new_indicators, new_sep)
    
    # Calcul des longueurs
    length_1 = []
    for seg in segs: 
        if seg.tolist():
             if seg[0] == 1:
                  length_1.append(len(seg))
    if not length_1:
         length_1.append(0)
    length_1 = np.array(length_1)
    
    n_segments = len(np.where(length_1 > duration*fs)[0])
    
    return n_segments


def mean_quadratic(values):
    
    mean_quad = 0
    for value in values:
        mean_quad += value**2 
        
    N = len(values)
    mean_quad = np.sqrt(1/N*mean_quad)
    
    return mean_quad


# New function for rsp used in siglife, class Breath, def analyze 
def compute_rsp_features_unwrap(sig, peaks, valleys, peaks_amps, valleys_amps, 
                                peaks_times, valleys_times, fs):
    # RR interval = breath to breath interval or peak to peak interval
    rr_intervals_s          = []
    cycles_times_s          = [] 
    inhale_intervals_s      = []
    exhale_intervals_s      = []
    inhale_amplitudes_s     = []
    exhale_amplitudes_s     = []
    
    # Inhale interval / Exhale interval ratio (temporal ratio)
    in_exhale_interval_ratio_s  = []
    
    # Inhale amplitude / Exhale amplitude ratio
    in_exhale_amplitude_ratio_s = []
    
    # For each segment/list i from the list of lists 
    for i in range(len(sig)):
        peaks_times_seg     = np.array(peaks_times[i])
        peaks_seg           = np.array(peaks[i])      
        valleys_seg         = np.array(valleys[i])
        peaks_amps_seg      = np.array(peaks_amps[i])
        valleys_amps_seg    = np.array(valleys_amps[i])
        
        cycles_times        = []
        rr_intervals        = []
        inhale_intervals    = []
        exhale_intervals    = []
        inhale_amplitudes   = []
        exhale_amplitudes   = []
        in_exhale_interval_ratio  = []
        in_exhale_amplitude_ratio = []
            
        if(len(peaks_seg) >= 2 and len(valleys_seg) >= 3):
            
            # Select segment when it starts with a valley and ends with a valley
            peaks_times_seg, peaks_seg, valleys_seg, peaks_amps_seg, \
                valleys_amps_seg = _valley_to_valley_rsp_segment(
                                    i, peaks_times_seg, peaks_seg, valleys_seg,
                                    peaks_amps_seg, valleys_amps_seg
                                    )
            # # View result
            # import matplotlib.pyplot as plt 
            # plt.figure()
            # plt.title("New peaks and valleys")
            # plt.plot(sig[i], color = 'black', label = 'sig cleaned')
            # plt.scatter(peaks_seg, peaks_amps_seg, color = 'r', label = 'peaks')
            # plt.scatter(valleys_seg, valleys_amps_seg, color = 'b', label = 'valleys')
            # plt.legend()
            
            # Compute rsp features
            rr_intervals, in_exhale_interval_ratio, in_exhale_amplitude_ratio,\
                inhale_intervals, exhale_intervals, inhale_amplitudes, \
                    exhale_amplitudes = _compute_rsp_features(
                                        peaks_seg, valleys_seg, 
                                        peaks_amps_seg, valleys_amps_seg, 
                                        fs)

            # cycles_times are the timestamps of of breath cycles, corresponding 
            # to peaks timestamps
            cycles_times = peaks_times[i][:-1]
        
        # Store inicators for each segment 
        cycles_times_s.append(cycles_times)
        rr_intervals_s.append(rr_intervals)
        
        inhale_intervals_s.append(inhale_intervals)
        exhale_intervals_s.append(exhale_intervals)
        
        inhale_amplitudes_s.append(inhale_amplitudes)
        exhale_amplitudes_s.append(exhale_amplitudes)
        
        in_exhale_interval_ratio_s.append(in_exhale_interval_ratio)
        in_exhale_amplitude_ratio_s.append(in_exhale_amplitude_ratio)
            
    # Construct the output
    output = {}
    output['rsp_cycles_times']  = cycles_times_s
    output['rsp_rr_intervals']  = rr_intervals_s
    
    output['inhale_intervals']  = inhale_intervals_s
    output['exhale_intervals']  = exhale_intervals_s
    
    output['inhale_amplitudes'] = inhale_amplitudes_s
    output['exhale_amplitudes'] = exhale_amplitudes_s
    
    output['inhale_exhale_interval_ratio']  = in_exhale_interval_ratio_s
    output['inhale_exhale_amplitude_ratio'] = in_exhale_amplitude_ratio_s
    
    return output

# --- Function used in compute_rsp_features_unwrap()---
def _valley_to_valley_rsp_segment(i, peaks_times, peaks, valleys, 
                                 peaks_amps, valleys_amps):
    
    # Select segment when it starts with a valley and ends with a valley
    # Chronologically there is: valley, peak, valley, peak, valley, etc
    # As result, for each segment there are total peaks = total valleys + 1
    # The idea is to have N completed breath cycles, 
    # inhale_intervals = exhale_intervals = cycles_times = rr_intarvels
    
    # 1) start with a peak, finish with a peak 
    if(valleys[0] > peaks[0] and valleys[-1] < peaks[-1]):
        peaks       = peaks[1:-1]
        peaks_amps  = peaks_amps[1:-1]
        peaks_times = peaks_times[1:-1]
        
    # 2) start with a peak, finish with a valley
    if( valleys[0] > peaks[0] and valleys[-1] > peaks[-1]):
        peaks       = peaks[1:]
        peaks_amps  = peaks_amps[1:]
        peaks_times = peaks_times[1:]
    
    # 3) start with a valley, finish with a peak
    if( valleys[0] < peaks[0] and valleys[-1] < peaks[-1]):
        peaks       = peaks[:-1]
        peaks_amps  = peaks_amps[:-1]
        peaks_times = peaks_times[:-1]
    
    # 4) start with a valley, finsh with a valley
    if( valleys[0] < peaks[0] and valleys[-1] > peaks[-1]):
        peaks_times = peaks_times[:]
        
    return peaks_times, peaks, valleys,\
           peaks_amps, valleys_amps 

# --- Function used in compute_rsp_features_unwrap() ---
def _compute_rsp_features(peaks, valleys, peaks_amps, valleys_amps, fs):
    
    # Compute valley to valley intervals = time taken to perform each cycle
    # unit of measurement  is seconds
    rr_intervals             = (valleys[1:] - valleys[:-1])/fs  
    
    # Compute intervals and ratio of inhale and exhale 
    inhale_intervals         = (peaks[:] - valleys[:-1])/fs                     # /fs to transform interval in seconds 
    exhale_intervals         = (valleys[1:] - peaks[:])/fs
    in_exhale_interval_ratio = inhale_intervals/exhale_intervals 
    
    # Compute amplitudes and ratio of inhale and exhale 
    inhale_amplitudes         = abs(peaks_amps[:] - valleys_amps[:-1])        
    exhale_amplitudes         = abs(valleys_amps[1:] - peaks_amps[:])   
    in_exhale_amplitude_ratio = inhale_amplitudes/exhale_amplitudes 
    
    return rr_intervals,in_exhale_interval_ratio, in_exhale_amplitude_ratio, \
        inhale_intervals, exhale_intervals, inhale_amplitudes, exhale_amplitudes

def compute_rr_features_unwrap(peaks_times, peaks, fs):
    
    times_rr_s          = []
    rr_s                = []
    times_rr_stats_s    = []
    
    sdnn_s              = []
    rmssd_s             = []
    lnrmssd_s           = []
    pnn50_s             = []
    rr_diff = []
    std_cons = []
    
    for i, peaks_seg in enumerate(peaks):
        peaks_seg   = np.array(peaks_seg)
        times_rr    = peaks_times[i][1:]
        # *1000 et non /1000 for  interval it in ms
        rr_interval = (peaks_seg[1:] - peaks_seg[:-1])/(fs)*1e3                
        rr_interval =np.array([ round(rs ,2) for rs in rr_interval])
        if len(rr_interval)>1:
            std_cons.append(np.std(rr_interval))
        
        if len(rr_interval) > 0:
            times_rr_s.extend(times_rr)
            rr_s.extend(rr_interval)
            
            times_rr_stats_s.append(times_rr[-1])
            
            rr_diff.extend(abs(rr_interval[1:] - rr_interval[:-1]))
                
    pnn50  = 0
    rr_diff = np.array(rr_diff)
    if len(rr_diff) > 0:
        pnn50       = len(rr_diff[rr_diff > 50])/(len(rr_diff)*100)
        pnn50_s.append(pnn50)
   
    if len(std_cons)>0:
        #print('std_cons', std_cons)
        sdnn_s.append(np.mean(std_cons))
        
    if len(rr_interval) > 0:
        rmssd_s.append(mean_quadratic(rr_interval))
        lnrmssd_s.append(np.log(mean_quadratic(rr_interval)))

    output                      = {}
    output['times_rr']          = times_rr_s
    output['rr']                = rr_s
    output['times_rr_stats']    = times_rr_stats_s
    
    output['sdnn']              = sdnn_s
    output['rmssd']             = rmssd_s
    output['lnrmssd']           = lnrmssd_s
    output['pnn50']             = pnn50_s
    
    return output


def detect_qt_interval(siglist,peaks_R, fs):
    qt_list = [detect_QT_interval(signal=siglist[sig], fs=fs, peaks_R= peaks_R[sig]) for sig in range(len(siglist))]
    onset_lists = [qt[0] for qt in qt_list]
    offset_lists = [qt[1] for qt in qt_list]
    
    # onset_lists= [detect_QT_interval(signal=sig, fs=fs)[0] for sig in siglist]
    # offset_lists= [detect_QT_interval(signal=sig, fs=fs)[1] for sig in siglist]
    return onset_lists, offset_lists

def from_index_to_time(index_list, time_sig):
    time_list = [time_sig[ind] for ind in index_list]
    return time_list

def qt_length(onset_times : list, offset_times : list) :  
    qt_deltas= [offset_times[i] - onset_times[i] for i in range( len(offset_times))]
    return qt_deltas

def compute_Framingham_QTc (Rpeak_list, QT_list):
    """ 
    verify that Rpeak_list is in times and not in indexes
    
    """
    Framingham_QTc_perseg = []
    for i in range(len(Rpeak_list)):
        rri = Rpeak_list[i]
        if len(rri)>1:
            
            if (QT_list[i][0].dtype != 'float64'):
                qtLi = np.array(QT_list[i])/np.timedelta64(1, 'ms')
            else :
                qtLi = np.array(QT_list[i]/1000)
            
            #print('qtLi', qtLi[0])
                
            rr_list = np.array([(rri[i+1] - rri[i]) for i in range(len(rri)-1)])
            #print( 'rr_list', rr_list[:5])
            rr_list = rr_list/np.timedelta64(1, 'ms')
            # print( 'rr_list', rr_list[:5])
            # print('rr_list', rr_list[0].astype(int))
            # print( 'qtLi', qtLi[:5])
            # print('qtLi', qtLi[0].astype(int))
            
            # verify that values are coherent
            # #print('qtLi[:3]', qtLi[:3])
            # qtLi     = [qti.astype(int) for qti in qtLi if 250<qti.astype(int)<750]
            # #print('rr[:3]', rr_list[:3])
            # rri      = [rr.astype(int) for rr in rr_list if 400<rr.astype(int)<2000] # from 30bpm to 150 bpm,
            # #T and P merge in high BPMs making reading QT difficult, thus we stop at 150 bpm
            
            qt_rr = [[qtLi[i],rr_list[i]] for i in range( min( len(qtLi), len(rr_list) ) ) \
                     if ((250<qtLi[i]<700) & (400<rr_list[i]<2100))]
            
            # print('len(qtLi), len(rr_list), len(qt_rr)', len(qtLi), len(rr_list), len(qt_rr))
            # print(qt_rr[:3])
            
            if len(qt_rr)>0:
                qtLi = [qtrr[0] for qtrr in qt_rr]
                rri = [qtrr[1] for qtrr in qt_rr]
            else :
                return np.nan, np.nan
            
            #calc medians and QTc
            #print('i, len(rri),len(qtLi)',   i, len(rri),len(qtLi))
            if (len(rri)*len(qtLi)) >0:
                rr_median = np.median(rri)
                QT_median = np.median(np.array(qtLi))
                Framingham =1000 *  (QT_median/1000 + 0.154*(1 - rr_median/1000))
                #print('rr_median', rr_median,'QT_median', QT_median,'Framingham',  Framingham )
                Framingham_QTc_perseg.append(round(Framingham, 0))
                
                #print(Framingham_QTc_perseg)       
    Framingham_QTc_med = np.median(Framingham_QTc_perseg)
    return Framingham_QTc_perseg, round(Framingham_QTc_med,0)

def compute_qt_times(ecg_sig_clean_list : list,
                               ecg_time_clean_list : list, 
                               fs : int):
    q_start_index_s         = []
    q_start_times_s         = []
    t_stop_index_s          = []
    t_stop_times_s          = []
    qt_length_s             = []
    qt_length_ms            = []
    qt_length_mean          = []
    qt_length_std           = []
    qt_c_framingham         = []
    qt_c_framingham_per_seg = []
    qt_length_med           = []
    qt_length_median_corrected = []
    qt_length_unwrap_int    = []
    
    
    peaks_R_list =  getPeaks_unwrap(sigs = ecg_sig_clean_list , fs = fs)
    
    #print(len(peaks_R_list))
    if len(peaks_R_list) >0:
            
        #detect qt indexes
        qTindexes = detect_qt_interval(ecg_sig_clean_list,peaks_R_list, fs)
        q_start_index_s = qTindexes[0]
        t_stop_index_s = qTindexes[1]
        
        #search corresponding timestamps
        q_start_times_s= [from_index_to_time( q_start_index_s[i], 
                                             ecg_time_clean_list[i]) 
                          for i in range(len(q_start_index_s)) ]
        
        t_stop_times_s= [from_index_to_time(t_stop_index_s[i], 
                                            ecg_time_clean_list[i]) 
                         for i in range(len(t_stop_index_s)) ]
           
        #calc time length of QT segment    
        qt_length_s = [qt_length(q_start_times_s[i], t_stop_times_s[i]) 
                       for i in range(len(q_start_times_s))]
        
        #unwrap qt length to calc mean, median and std
        qt_length_unwrap = []
        qt_length_ms    = []
        for qt in qt_length_s :
            if qt != []:
                qt_length_unwrap        = qt_length_unwrap + qt   
                qt_length_ms.append(np.array(qt/np.timedelta64(1, 'ms')).astype(float))

        if len(qt_length_unwrap)>0:
            qt_length_unwrap            = np.array(qt_length_unwrap)/np.timedelta64(1, 'ms')
        qt_length_unwrap_int        = [qti.astype('int') for qti in qt_length_unwrap]
        qt_length_unwrap_int        = [round(qti,0) for qti in qt_length_unwrap_int]
        qt_length_ms                = np.array(qt_length_ms)
        if len(qt_length_unwrap_int)>0:
            qt_length_mean              = round(sum(qt_length_unwrap_int)/(len(qt_length_unwrap_int)),0)
        qt_length_med               = np.median(qt_length_unwrap_int)
        qt_length_std               = round(np.std(qt_length_unwrap_int),0)
        
        #calc qt corrected by rr
        qt_c_framingham_per_seg, qt_c_framingham  =  compute_Framingham_QTc (Rpeak_list = q_start_times_s, 
                                                                     QT_list = qt_length_s)                     
    if (len([qti for qti in qt_length_unwrap_int if 350<qti<550]))>0:
        qt_length_median_corrected         = np.median([qti for qti in qt_length_unwrap_int if 350<qti<550])
 
        
    output                                = {}
    
    output['q_start_index_s']             = q_start_index_s
    output['q_start_times_s']             = q_start_times_s
    
    output['t_stop_index_s']              = t_stop_index_s
    output['t_stop_times_s']              = t_stop_times_s
    
    output['qt_length']                   = qt_length_ms
    output['qt_length_unwrap']            = qt_length_unwrap_int
    output['qt_length_mean']              = qt_length_mean
    output['qt_length_std']               = qt_length_std
    output['qt_length_median']            = qt_length_med
    output['qt_length_median_corrected']  = qt_length_median_corrected
    output['qt_c_framingham']             = qt_c_framingham
    output['qt_c_framingham_per_seg']     = qt_c_framingham_per_seg
    
    return output

def get_peaks_times_info(times, sig, fs, peaks_times):
    
    peaks           = []
    rr_intervals    = []
    pnn50s          = []
    for id_seg, seg in enumerate(sig):
        times_seg   = times[id_seg]
        peak        = get_peaks_from_peaks_times(times_seg, peaks_times)
        peak        = np.array(peak)
        rr_interval = (peak[1:] - peak[:-1])/fs*1e3 
        rr_diff     = abs(rr_interval[1:] - rr_interval[:-1])
        pnn50       = len(rr_diff[rr_diff > 50])/len(rr_diff)*100
        
        rr_intervals.append(rr_interval) 
        peaks.append(peak)
        pnn50s.append(pnn50)
    
    output                  = {}
    output['peaks']         = np.array(peaks)
    output['rr_intervals']  = np.array(rr_intervals)
    output['pnn50s']        = np.array(pnn50s)
    
    return output

def get_peaks_times_unwrap(times, peaks):
    peaks_times = []    
    for i, t in enumerate(times):
        t = np.array(t)
        p = peaks[i]        
        peaks_time = get_peaks_times(t, p)
        peaks_times.append(peaks_time)

    return peaks_times

def get_peaks_times(times, peaks):

    peaks_times = []    
    times = np.array(times)
    if len(peaks) > 0:
        peaks_times = times[peaks]

    return peaks_times

def get_peaks_amps_unwrap(sig, peaks):

    peaks_amps = []    
    for i, seg in enumerate(sig):
        seg = np.array(seg)
        p = peaks[i]        
        peaks_amp = get_peaks_amps(seg, p)
        peaks_amps.append(peaks_amp)

    return peaks_amps

def get_peaks_amps(sig, peaks):
   
    sig = np.array(sig)
    if len(peaks) > 0:
        amp = sig[peaks]      
    return amp

def next_pow_2(x):
    
    count = 0
    while x >= 1:
        x = x / 2
        count += 1
    return count

def averaged_filter(sig, window):
    for j in range(window, len(sig)-window):
        sig[j] = np.mean(sig[j-window:j+window])
    return sig

def error_relative(a, b):
    return (a - b)/a

def fw_version_test(fw_version):
    
    output = {}
    output['error'] = False
    if not isinstance(fw_version, str):
        output['error'] = 'is not str'
        return output
        
    fw = fw_version.split('_')
    if len(fw) != 2:
        output['error'] = 'split by "_" failed'
        return output
    
    try:
        fw = np.array(fw, dtype='int')
    except:
        output['error'] = 'conversion in int failed'
        return output
        
    card_version    = fw[0]
    hard_version    = fw[1]
    
    if card_version <= 0:
        output['error'] = 'card_version is not positive'
        return output
        
    if hard_version <= 0:
        output['error'] = 'fw_version is not positive'
        return output
    
    output['card_version'] = card_version 
    output['hard_version'] = hard_version

    return output

# def add_title(slide, text, fontsize=36, bold=True, color=None):
#     if color is None:
#         color = RGBColor(0x00, 0x00, 0x00)
        
#     # creating textBox
#     title_left          = title_width = title_height = Inches(1)
#     title_top           = 0
#     txBox               = slide.shapes.add_textbox(title_left, title_top, title_width, title_height)
      
#     # creating textFrames
#     tf                  = txBox.text_frame
#     p                   = tf.add_paragraph() 
#     p.text              = text
#     p.font.bold         = bold
#     p.font.size         = Pt(fontsize)
#     p.font.color.rgb    = color
    
# def add_images(ppt, slide, path_save, img_name1, img_name2):
#     path_save1   = path_save + img_name1
#     path_save2   = path_save + img_name2
#     img_left    = 1*1e5
#     img_top     = 15*1e5
#     img_width   = 60*1e5
#     img_height  = 40*1e5
#     img_left2   = img_width + 1*1e5
#     pic         = slide.shapes.add_picture(path_save1, img_left, img_top, img_width, img_height)
#     pic         = slide.shapes.add_picture(path_save2, img_left2, img_top, img_width, img_height)

# def add_slide_1(template_path, ppt, slide_layout, info, fontsize=36, bold=True, color=None):
#     if color is None:
#         color = RGBColor(0x00, 0x00, 0x00)
    
        
#     slide   = ppt.slides.add_slide(slide_layout)
    
#     img_left    = 1*1e5
#     img_top     = 1*1e5
#     img_width   = 25*1e5
#     img_height  = 25*1e5
#     img_left2   = img_width + 1*1e5
#     pic         = slide.shapes.add_picture(template_path + 'Chronolife_logo.jpg', 
#                                            img_left, img_top, img_width, img_height)
    
#     # creating textBox
#     left    = Cm(3)
#     top     = Cm(7)
#     width   = Cm(27)
#     height  = Cm(1)
#     txBox               = slide.shapes.add_textbox(left, top, width, height)
#     tf                  = txBox.text_frame
#     p                   = tf.add_paragraph() 
#     p.text              = info['main_title']
#     p.font.bold         = bold
#     p.font.size         = Pt(fontsize)
#     p.font.color.rgb    = color
#     p.alignment         = PP_ALIGN.CENTER
    
#     p                   = tf.add_paragraph() 
#     p.text              = info['second_title']
#     p.font.size         = Pt(22)
#     p.alignment         = PP_ALIGN.CENTER
    
#     # Operator
#     left    = Cm(1)
#     top     = Cm(15)
#     width   = Cm(3)
#     height  = Cm(1)
#     txBox               = slide.shapes.add_textbox(left, top, width, height)
#     tf                  = txBox.text_frame
#     tf.text             = info['operator']
    
#     left    = Cm(29)
#     top     = Cm(15)
#     width   = Cm(3)
#     height  = Cm(1)
#     txBox               = slide.shapes.add_textbox(left, top, width, height)
#     tf                  = txBox.text_frame
#     tf.text             = info['date']
    
    
# def add_slide_info(ppt, slide_layout, title, info, color=None):
#     if color is None:
#         color = RGBColor(0x00, 0x00, 0x00)
    
#     slide   = ppt.slides.add_slide(slide_layout)
#     add_title(slide, 'Global info', fontsize=36, bold=True, color=color)
    
#     # Adjusting the width !  
#     x, y, cx, cy = Inches(2), Inches(2), Inches(4), Inches(1.5) 

#     # Adding tables
#     nrows = 5
#     ncols = 2
#     shape = slide.shapes.add_table(nrows, ncols, x, y, cx, cy)
    
#     #  table header padding 
#     shape.table.cell(0, 0).text = 'Info'
#     shape.table.cell(1, 0).text = 'User id'
#     shape.table.cell(2, 0).text = 'Start'
#     shape.table.cell(3, 0).text = 'Stop'
#     shape.table.cell(4, 0).text = 'Record duration'
    
#     #  table content filling 
#     shape.table.cell(0, 1).text = ''
#     shape.table.cell(1, 1).text = info['user']
#     shape.table.cell(2, 1).text = info['from_time']
#     shape.table.cell(3, 1).text = info['to_time']
#     shape.table.cell(4, 1).text = info['record_duration']
    
# def add_slide_image(ppt, slide_layout, path_save, title, img_name1, img_name2, color=None):
#     if color is None:
#         color = RGBColor(0x00, 0x00, 0x00)
    
#     slide   = ppt.slides.add_slide(slide_layout)
#     add_title(slide,  title, fontsize=36, bold=False, color=color)
#     add_images(ppt, slide, path_save, img_name1, img_name2)
    
# def delete_slide(ppt, index):
#     xml_slides = ppt.slides._sldIdLst  # pylint: disable=W0212
#     slides = list(xml_slides)
#     xml_slides.remove(slides[index])
    
# def ppt_report(template_path, path_save, info, flag_temp=False):
    
#     ppt = Presentation(template_path + 'template_python.pptx') 
    
#     # Create blank slide layout
#     slide_layout = ppt.slide_layouts[6]
    
#     color = RGBColor(0x46, 0x94, 0xAF)
#     add_slide_1(template_path, ppt, slide_layout, info, fontsize=36, bold=True, color=color)
    
#     # SLIDE TABLE INFO
#     add_slide_info(ppt, slide_layout, 'Global info', info, color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Global info', 
#               img_name1='global_disconnection_bar.png', 
#               img_name2='global_usable_bar.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Activity', 
#               img_name1='result_mean_activity_bar.png', 
#               img_name2='result_n_steps_bar.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Activity distributions', 
#               img_name1='result_mean_activity_dist.png', 
#               img_name2='result_n_steps_dist.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Heart rate', 
#               img_name1='result_bpm_bar.png', 
#               img_name2='result_hrv_bar.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Heart rate distributions', 
#               img_name1='result_bpm_dist.png', 
#               img_name2='result_hrv_dist.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiratory rate', 
#               img_name1='result_rpm_abd_bar.png', 
#               img_name2='result_rpm_tho_bar.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiratory rate distributions', 
#               img_name1='result_rpm_abd_dist.png', 
#               img_name2='result_rpm_tho_dist.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Temperature', 
#               img_name1='result_temp_left_bar.png', 
#               img_name2='result_temp_right_bar.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Temperature distributions', 
#               img_name1='result_temp_left_dist.png', 
#               img_name2='result_temp_right_dist.png', color=color)
    
#     if flag_temp:
#         add_slide_image(ppt, slide_layout, path_save, title='Temperatures', 
#                   img_name1='temp_2_median.png', 
#                   img_name2='temp_1_median.png', color=color)
        
#         add_slide_image(ppt, slide_layout, path_save, title='Temperatures variations', 
#                   img_name1='temp_2_var_median.png', 
#                   img_name2='temp_1_var_median.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='ECG usable - examples', 
#               img_name1='ecg_random_seg0.png', 
#               img_name2='ecg_random_seg1.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='ECG usable - examples', 
#               img_name1='ecg_random_seg2.png', 
#               img_name2='ecg_random_seg3.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='ECG usable - examples', 
#               img_name1='ecg_random_seg4.png', 
#               img_name2='ecg_random_seg5.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiration thoracic - usable examples', 
#               img_name1='breath_1_random_seg0.png', 
#               img_name2='breath_1_random_seg1.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiration thoracic - usable examples', 
#               img_name1='breath_1_random_seg2.png', 
#               img_name2='breath_1_random_seg3.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiration thoracic - usable examples', 
#               img_name1='breath_1_random_seg4.png', 
#               img_name2='breath_1_random_seg5.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiration abdominal - usable examples', 
#               img_name1='breath_2_random_seg0.png', 
#               img_name2='breath_2_random_seg1.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiration abdominal - usable examples', 
#               img_name1='breath_2_random_seg2.png', 
#               img_name2='breath_2_random_seg3.png', color=color)
    
#     add_slide_image(ppt, slide_layout, path_save, title='Respiration abdominal - usable examples', 
#               img_name1='breath_2_random_seg4.png', 
#               img_name2='breath_2_random_seg5.png', color=color)
    
#     delete_slide(ppt, 0)
    
#     ppt.save(path_save + info['user'] + '_ppt_report.pptx')
      
#     print('--------------------------------------------------------')
#     print("PPT REPORT")
#     print(info['user'] + '_ppt_report.pptx saved in:', path_save.replace('/', '\\').replace('\\\\', '\\'))
    
def get_peaks_clean(times_clean, sig_clean, peaks_times):
    
    if len(peaks_times)==0:
        return [], []
    
    imin                = np.where(peaks_times >= times_clean[0])[0]
    imax                = np.where(peaks_times <= times_clean[-1])[0]
    
    if len(imin) == 0 or len(imax) == 0:
        return [], []
    imin = imin[0]
    imax = imax[-1]
    peaks_times_clean   = peaks_times[imin:imax]
    
    peaks_clean = []
    for i in range(len(peaks_times_clean)):
        peaks_clean.append(np.where(times_clean == peaks_times_clean[i])[0][0])

    return peaks_times_clean, peaks_clean 

# used in get_rsp_peaks_clean_unwrap
def get_rsp_peaks_clean(times_clean, sig_clean, peaks_times):
    
    if len(peaks_times)==0:
        return [], []
    
    imin = np.where(peaks_times >= times_clean[0])[0]
    imax = np.where(peaks_times <= times_clean[-1])[0]
    
    if len(imin) == 0 or len(imax) == 0:
        return [], []
    imin = imin[0]
    imax = imax[-1]
    peaks_times_clean = peaks_times[imin:imax+1]                                # Change: [imin:imax] -> peaks_times[imin:imax+1] 
    
    peaks_clean = []
    for i in range(len(peaks_times_clean)):
        peaks_clean.append(np.where(times_clean == peaks_times_clean[i])[0][0])

    return peaks_times_clean, peaks_clean 

# Used in Siglige by Breath 
def get_rsp_peaks_clean_unwrap(times_clean, sig_clean, peaks_times):
    
    peaks_times_clean_s = []
    peaks_amps_clean_s  = []
    peaks_clean_s       = []
    
    if is_list_of_list(peaks_times):
        peaks_times = unwrap(peaks_times)
    
    # For each segment 
    for i in range(len(times_clean)):
        times_seg_clean = times_clean[i]
        seg_clean = sig_clean[i]
        
        peaks_times_clean, peaks_clean = get_rsp_peaks_clean(times_seg_clean, 
                                                         seg_clean, 
                                                         peaks_times)
        peaks_amps_clean = np.array(seg_clean)[peaks_clean]
        
        peaks_times_clean_s.append(peaks_times_clean)
        peaks_amps_clean_s.append(peaks_amps_clean)
        peaks_clean_s.append(peaks_clean)
        
    return peaks_times_clean_s, peaks_amps_clean_s, peaks_clean_s 

def get_peaks_clean_unwrap(times_clean, sig_clean, peaks_times):
    
    peaks_times_clean_s = []
    peaks_clean_s       = []
    
    if is_list_of_list(peaks_times):
        peaks_times = unwrap(peaks_times)
            
    for i in range(len(times_clean)):
        times_seg_clean         = times_clean[i]
        seg_clean               = sig_clean[i]
        
        peaks_times_clean, peaks_clean = get_peaks_clean(times_seg_clean, 
                                                         seg_clean, 
                                                         peaks_times)
        
        peaks_times_clean_s.append(peaks_times_clean)
        peaks_clean_s.append(peaks_clean)
    
    return peaks_times_clean_s, peaks_clean_s 

def compute_mean_std_window_unwrap(times, sig, fs, window_time):
    """ Compute mean per minute

        Parameters
        ---------------------------------
        sig:            signal
        fs:             sampling frequency
        window_time:    Compute hr per minute every window seconds

        Returns
        ---------------------------------
        mean:       mean of rate
        std:        std of rate
        times_mean: timestamps of rate computing

    """
    means_s       = []
    stds_s        = []
    times_mean_s  = []
    
    for i, seg in enumerate(sig):
        times_seg = times[i]
        means, stds, times_mean = compute_mean_std_window(times_seg, seg, 
                                                         fs, window_time)
        means_s.append(means)
        stds_s.append(stds)
        times_mean_s.append(times_mean)
        
    return np.array(means_s), np.array(stds_s), np.array(times_mean_s)

def compute_mean_std_window(times, sig, fs, window_time):
    """ Compute mean per window time

        Parameters
        ---------------------------------
        sig:            signal
        fs:             sampling frequency
        window_time:    Compute hr per minute every window seconds

        Returns
        ---------------------------------
        means:      mean of rate
        stds:       std of rate
        times_var:  timestamps of rate computing

    """
    means       = []
    stds        = []
    times_mean  = []
    
    window = window_time*fs
    for i in range(0, len(sig), window):
        imin = i
        imax = imin + window
        
        if imax >= len(sig):
            imax = len(sig) - 1
        
        seg = sig[imin:imax]
        
        means.append(np.mean(seg))
        stds.append(np.std(seg))
        times_mean.append(times[imax])
    
    if window > len(sig):
        means.append(np.mean(sig))
        stds.append(np.std(sig))
        times_mean.append(times[-1])
        
    return np.array(means), np.array(stds), np.array(times_mean)

def compute_median_iqr_window_unwrap(times, sig, fs, window_time):
    """ Compute median per minute

        Parameters
        ---------------------------------
        sig:            signal
        fs:             sampling frequency
        window_time:    Compute hr per minute every window seconds

        Returns
        ---------------------------------
        median:         mean of rate
        iqr:            iqr of rate
        times_mean:     timestamps of rate computing

    """
    medians_s       = []
    iqrs_s          = []
    times_median_s  = []
    
    for i, seg in enumerate(sig):
        times_seg = times[i]
        medians, iqrs, times_median = compute_median_iqr_window(times_seg, seg, 
                                                         fs, window_time)
        medians_s.extend(medians)
        iqrs_s.extend(iqrs)
        times_median_s.extend(times_median)
        
    return np.array(medians_s), np.array(iqrs_s), np.array(times_median_s)

def compute_median_iqr_window(times, sig, fs, window_time):
    """ Compute median per window time

        Parameters
        ---------------------------------
        sig:            signal
        fs:             sampling frequency
        window_time:    Compute hr per minute every window seconds

        Returns
        ---------------------------------
        means:      mean of rate
        stds:       std of rate
        times_var:  timestamps of rate computing

    """
    medians         = []
    iqrs            = []
    times_median    = []
    
    window = window_time*fs
    for i in range(0, len(sig), window):
        imin = i
        imax = imin + window
        
        if imax >= len(sig):
            imax = len(sig) - 1
        
        seg = sig[imin:imax]
        
        if len(seg) > 1:
            medians.append(np.median(seg))
            iqrs.append(np.percentile(seg, 75)-np.percentile(seg, 25))
            times_median.append(times[imax])
        elif len(seg) == 1:
            medians.append(seg[0])
            iqrs.append(0)
            times_median.append(times[imax])
    
    if window > len(sig) and len(sig) > 0:
        medians.append(np.median(sig))
        iqrs.append(np.percentile(sig, 75)-np.percentile(sig, 25))
        times_median.append(times[-1])
        
    return medians, iqrs, times_median