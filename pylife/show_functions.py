# -*- coding: utf-8 -*-
import math
import numpy as np
from pylife.env import get_env
# --- Add imports for PROD and DEV env
from pylife.useful import is_list_of_list
from pylife.useful import unwrap
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse
    import seaborn as sns
    import pandas as pd

def show_indicators(times, sig, indicators, color=None, amp=None):
    
    if amp is None:
        amp = np.max(sig)*1.2
    indicators = np.array(indicators)
    
    labels = plt.legend(fontsize=16).get_texts()
        
    if len(indicators) > 0:
        # if is_list_of_list(indicators):
        #     for id_seg, seg in enumerate(sig):
        #         times_seg = times[id_seg]
        #         indicators_seg = indicators[id_seg]
        #         indicators_seg = np.array(indicators_seg)
        #         g, = plt.plot(times_seg, seg, label='Signal')
        #         if color is not None:
        #             plt.fill_between(times_seg,
        #                              (indicators_seg*amp).tolist(),
        #                              alpha=0.3, label='Indicators',
        #                              color=color)
        #         else:
        #             plt.fill_between(times_seg, (indicators_seg*amp).tolist(),
        #                              alpha=0.3, label='Indicators')
        # else:
        if len(labels) == 0:
            g, = plt.plot(times, sig, label='Signal', c=color)
        else:
            g, = plt.plot(times, sig, c=color)

        if color is not None:
            if len(labels) == 0:
                plt.fill_between(times, (indicators*amp).tolist(), alpha=0.3,
                                 label='Indicators', color=color)
            else:
                plt.fill_between(times, (indicators*amp).tolist(), alpha=0.3, color=color)
        else:
            if len(labels) == 0:
                plt.fill_between(times, (indicators*amp).tolist(), alpha=0.3, label='Indicators')
            else:
                plt.fill_between(times, (indicators*amp).tolist(), alpha=0.3)
                
    plt.legend(fontsize=16)
                
def poincare_plot(RR):
    """ Plot Poincaré plot with
    Inputs: list des rr intervals consécutifs
    """
    ax1 = []
    ax2 = []
    RR_stat = []
    diff = []
    if is_list_of_list(RR):
        for rr in RR:
            ax1.extend(rr[:-1])
            ax2.extend(rr[1:])
            RR_stat.extend(rr)
            diff.extend((np.array(rr[1:])-np.array(rr[:-1]))**2)
    else:
        ax1.extend(RR[:-1])
        ax2.extend(RR[1:])
        RR_stat.extend(RR)
        diff.extend((np.array(RR[1:])-np.array(RR[:-1]))**2)

    ax1 = np.array(ax1)
    ax2 = np.array(ax2)

    SDSD = math.sqrt(np.mean(diff))
    SDRR = np.std(RR_stat)
    SD1 = (1/np.sqrt(2)) * SDSD  # width of poincare cloud
    SD2 = np.sqrt((2 * SDRR ** 2) - (0.5 * SDSD ** 2))  # length of the cloud

    plt.figure()
    plt.scatter(ax1, ax2, s=12)
    plt.xlim(0, 2.1)
    plt.ylim(0, 2.1)
    np.mean(RR_stat)
    plt.annotate('', xytext=(0.3, 0.3), xy=(3/2*np.mean(RR_stat),
                                            3/2*np.mean(RR_stat)),
                 arrowprops=dict(facecolor='black',
                                 arrowstyle='->'))
    plt.annotate('', xytext=(1.5, - 1.5 + 2 * np.mean(RR_stat)),
                 xy=(0.4, - 0.4 + 2 * np.mean(RR_stat)),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.text(3/2 * np.mean(RR_stat), 3/2 * np.mean(RR_stat), 'SD2')
    plt.text(0.4, - 0.4 + 2*np.mean(RR_stat), 'SD1')
    ax = plt.gca()

    ellipse = Ellipse(xy=(np.mean(RR_stat), np.mean(RR_stat)), width=2*SD2,
                      height=2*SD1, angle=45,
                      edgecolor='red', fc='None', lw=2)
    ax.add_patch(ellipse)

    plt.xlabel('RR_n (s)')
    plt.ylabel('RR_n+1 (s)')
    plt.title("Poincare plot")
    plt.show()


def ppm_distplot(ppm):
    ppm = np.array(ppm)
    sns.distplot(ppm)
    plt.axvline(ppm.mean())
    plt.xlabel('Peaks per minute', fontsize=18)
    plt.ylabel('Distribution', fontsize=18)
    plt.title(('PPM distribution (mean: %.0f PPM)' % ppm.mean()), fontsize=18)
    
def show_result_plot(df, row, col, key, ylim, fontsize, yoffset, ytype, hue=None, flag_legend=False, disp_minmax=True):
    
    if hue is not None:
        df['hue'] = hue
    
    df = df[df[col].isnull() == False]
    df = df[df[col] != 'nan']
    df[col][df[col] == 'None'] = 0 
    sig_colors = get_sig_colors()
    color_low = sig_colors[key][1]
    color_high = sig_colors[key][0]
    colors = np.repeat(color_low, len(df))
    
    if hue is not None:
        colors = []
        for hu in df['hue']:
            if hu: 
                colors.append(color_high)
            else:
                colors.append(color_low)
    plt.plot(df[row], df[col].astype('float'), c='grey')
    for i in range(len(colors)):
        plt.plot(df[row].iloc[i], df[col].astype('float').iloc[i], marker='o', 
                 linestyle='', c=colors[i], linewidth=2)
    plt.grid('on')
    plt.ylim(ylim)
    plt.title(col, fontsize=fontsize)
    plt.xlabel('')
    xticks = []
    step = int(np.floor(len(df[row])/10))
    if step == 0:
        step = 1
    iold = -step
    for i in range(0, len(df[row])):
        if i-iold == step:
            xticks.append(df[row].values[i][11:-3])
            iold = i
        else:
            xticks.append('')
    plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=8, rotation=45)
    plt.yticks(fontsize=14)        
    
    if hue is not None and flag_legend:
        leg1 = mpatches.Patch(color=color_low, label='Rest')
        leg2 = mpatches.Patch(color=color_high, label='Activity')
        plt.legend(handles=[leg1, leg2], fontsize=fontsize)
    
    if disp_minmax:
        values = np.array(df[col].astype('float').values)
        values_not_nan = values[np.isnan(values) == False]
        if len(values_not_nan) == 0:
            return
        y = np.max(values_not_nan)
        x = np.where(values == y)[0][0]
        if y == 'nan':        
            plt.text(x-.1, 0, '', fontsize=fontsize)
        else:
            plt.text(x-.1, y+yoffset, str(y.astype(ytype)), fontsize=fontsize)
        
        y = np.min(values_not_nan)
        x = np.where(values == y)[0][0]
        if y == 'nan':        
            plt.text(x-.1, 0, '', fontsize=fontsize)
        else:
            plt.text(x-.1, y+yoffset, str(y.astype(ytype)), fontsize=fontsize)
   
        
def show_result_bar(df, row, col, key, ylim, fontsize, yoffset, ytype, hue=None, flag_legend=False, disp_minmax=True):

    if hue is not None:
        df['hue'] = hue
    df = df[df[col].isnull() == False]
    df = df[df[col] != 'nan']
    df[col][df[col] == 'None'] = 0
    sig_colors = get_sig_colors()
    color_low = sig_colors[key][1]
    color_high = sig_colors[key][0]
    colors = np.repeat(color_low, len(df))
    if hue is not None:
        colors = []
        for hu in hue:
            if hu: 
                colors.append(color_high)
            else:
                colors.append(color_low)
    
    # if key != 'mean_activity':
        # plt.subplot(4,1,(1,3))
    plt.bar(df[row], df[col].astype('float'), zorder=3, color=colors)
    plt.grid(zorder=0, axis='y')
    plt.ylim(ylim)
    plt.title(col, fontsize=fontsize)
    plt.xlabel('')
    xticks = []
    step = int(np.floor(len(df[row])/10))
    if step == 0:
        step = 1
    iold = -step
    for i in range(0, len(df[row])):
        if i-iold == step:
            xticks.append(df[row].values[i][11:-3])
            iold = i
        else:
            xticks.append('')
    plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=8, rotation=45)
    plt.yticks(fontsize=14)        
    
    if hue is not None and flag_legend:
        leg1 = mpatches.Patch(color=color_low, label='Rest')
        leg2 = mpatches.Patch(color=color_high, label='Activity')
        plt.legend(handles=[leg1, leg2], fontsize=fontsize)
    
    if disp_minmax:
        values = np.array(df[col].astype('float').values)
        values_not_nan = values[np.isnan(values) == False]
        if len(values_not_nan) == 0:
            return
        y = np.max(values_not_nan)
        x = np.where(values == y)[0][0]
        if y == 'nan':        
            plt.text(x-.1, 0, '', fontsize=fontsize)
        else:
            plt.text(x-.1, y+yoffset, str(y.astype(ytype)), fontsize=fontsize)
        
        # y = np.min(values_not_nan)
        # x = np.where(values == y)[0][0]
        # if y == 'nan':        
        #     plt.text(x-.1, 0, '', fontsize=fontsize)
        # else:
        #     plt.text(x-.1, y+yoffset, str(y.astype(ytype)), fontsize=fontsize)
    
    # if key == 'mean_activity':
    #     return
    # col = 'result'
    # key = 'mean_activity'
    # if hue is not None:
    #     df_mean_acc['hue'] = hue
    # df_mean_acc = df_mean_acc[df_mean_acc[col].isnull() == False]
    # df_mean_acc = df_mean_acc[df_mean_acc[col] != 'nan']
    # df_mean_acc[col][df_mean_acc[col] == 'None'] = 0
    # plt.subplot(4,1,4)
    # sig_colors = get_sig_colors()
    # color_low = sig_colors[key][1]
    # color_high = sig_colors[key][0]
    # colors = np.repeat(color_low, len(df_mean_acc))
    # if hue is not None:
    #     colors = []
    #     for hu in hue:
    #         if hu: 
    #             colors.append(color_high)
    #         else:
    #             colors.append(color_low)
    # plt.bar(df_mean_acc[row], df_mean_acc[col].astype('float'), zorder=3, color=colors)
    # plt.grid(zorder=0, axis='y')
    # ylim = get_sig_ranges()[key]
    # plt.ylim(ylim)
    # plt.xlabel('')
    # plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=8, rotation=45)
    # # plt.yticks(fontsize=14)        
    
    # if hue is not None and flag_legend:
    #     leg1 = mpatches.Patch(color=color_low, label='Activity Level')
    #     plt.legend(handles=[leg1], fontsize=fontsize)
    
    # plt.subplot(4,1,(1,3))
    # plt.xticks([])
         
def show_result_dist(df, col, key, xlim, hue=None, fontsize=10):
    
    if hue is not None:
        df['hue'] = hue
    df = df[df[col].isnull() == False]
    df = df[df[col] != 'nan']
    df_clear = df[df[col] != 'None']
    df[col][df[col] == 'None'] = 0 
    sig_colors = get_sig_colors()
    color_low = sig_colors[key][1]
    color_high = sig_colors[key][0]
    if hue is None:
        median = None
        color = color_high
        df[col] = df[col].astype('float') 
        sns.distplot(df[col], hist=True, color=color_low)
        
        if len(df_clear) > 0:
            median = round(df_clear[col].astype('float').median()*10)/10
            plt.axvline(x=median, linestyle='--', color=color)
        leg1 = mpatches.Patch(color=color, label='median: ' + str(median))
        plt.legend(handles=[leg1], fontsize=fontsize)
    else:
        median1 = None
        median2 = None
        df1 = df[df['hue'] == False]
        df2 = df[df['hue'] == True]
        df1 = df1[df1[col].isnull() == False]
        df2 = df2[df2[col].isnull() == False]
        df1 = df1[df1[col] != 'nan']
        df2 = df2[df2[col] != 'nan']
        df_clear1 = df1[df1[col] != 'None']
        df1[col][df1[col] == 'None'] = 0 
        df_clear2 = df2[df2[col] != 'None']
        df2[col][df2[col] == 'None'] = 0 
        df1[col] = df1[col].astype('float') 
        df2[col] = df2[col].astype('float') 
        sns.distplot(df1[col], hist=True, color=color_low)
        sns.distplot(df2[col], hist=True, color=color_high)
        
        if len(df_clear1) > 0:
            median1 = round(df_clear1[col].median()*10)/10
            plt.axvline(x=median1, linestyle='--', color=color_low)
        if len(df_clear2) > 0:
            median2 = round(df_clear2[col].median()*10)/10
            plt.axvline(x=median2, linestyle='--', color=color_high)
        leg1 = mpatches.Patch(color=color_low, label='Rest median: ' + str(median1))
        leg2 = mpatches.Patch(color=color_high, label='Activity median: ' + str(median2))
        plt.legend(handles=[leg1,leg2], fontsize=fontsize)
        
    plt.xlabel(col + ' values', fontsize=fontsize)
    plt.title(col + ' distribution', fontsize=fontsize)
    plt.xlim(xlim)
    plt.yticks([])
    plt.xticks(fontsize=fontsize)
    
def get_sig_colors():
    cm = plt.cm.get_cmap('tab20').colors
    
    # acc_dark        = 'purple'
    # acc_light       = 'purple'
    
    # breath_dark     = tuple(np.array([104, 168, 0])/255)
    # breath_light    = tuple(np.array([135, 218, 0])/255)
    
    # ecg_dark        = tuple(np.array([62, 116, 146])/255)
    # ecg_light       = tuple(np.array([120, 170, 198])/255)
    
    temp_dark       = tuple(np.array([241, 134, 35])/255)
    temp_light      = tuple(np.array([247, 189, 137])/255)
    
    # acc_colors              = [acc_dark, acc_light]
    # breath_colors           = [breath_dark, breath_light]
    # ecg_colors              = [ecg_dark, ecg_light]
    # temp_colors             = [temp_dark, temp_light]
    # imp_colors              = [imp_dark, imp_light]
    
    acc_colors              = [cm[8], cm[9]]
    breath_colors           = [cm[4], cm[5]]
    ecg_colors              = [cm[18], cm[19]]
    temp_colors             = [cm[2], cm[3]]
    imp_colors              = [cm[10], cm[11]]
    # ecg_breath_colors       = [cm[11], cm[10]]
    # ecg_temp_colors         = [cm[13], cm[12]]
    # ecg_breath_temp_colors  = [cm[15], cm[14]]
    
    # acc_colors              = [cm[15], cm[1]]
    # breath_colors           = [cm[15], cm[5]]
    # ecg_colors              = [cm[15], cm[19]]
    # temp_colors             = [cm[15], cm[3]]
    # imp_colors              = [cm[15], cm[11]]
    
    
    sig_color = {'acc':                 acc_colors,
                 'accx':                acc_colors, 
                 'accy':                acc_colors, 
                 'accz':                acc_colors, 
                 'n_steps':             acc_colors,
                 'steps':               acc_colors,
                 'mean_activity':       acc_colors,
                 'mean_activity_level': acc_colors,
                 'breath':              breath_colors, 
                 'breath_1':            breath_colors, 
                 'breath_2':            breath_colors,
                 'rpm_tho':             breath_colors,
                 'rpm_abd':             breath_colors,
                 'rpm_1':               breath_colors,
                 'rpm_2':               breath_colors,
                 'amp_tho':             breath_colors,
                 'amp_abd':             breath_colors,
                 'ecg':                 ecg_colors, 
                 'bpm':                 ecg_colors, 
                 'hrv':                 ecg_colors, 
                 'pnn50':               ecg_colors, 
                 'amp_ecg':             ecg_colors,
                 'temp':                temp_colors, 
                 'temp_1':              temp_colors, 
                 'temp_2':              temp_colors, 
                 'temp_right':          temp_colors,
                 'temp_left':           temp_colors,
                 'imp':                 imp_colors,
                 'imp_1':               imp_colors,
                 'imp_2':               imp_colors,
                 'imp_3':               imp_colors,
                 'imp_4':               imp_colors,
                 # 'ecg_breath':          ecg_breath_colors,
                 # 'ecg_temp':            ecg_temp_colors,
                 # 'ecg_breath_temp':     ecg_breath_temp_colors,
                 }
    
    return sig_color

def get_summary_keys():
    keys = ['accx', 'breath_1', 'breath_2', 'ecg', 
            'temp_1', 'temp_2', 'imp_1', 'imp_2', 'imp_3', 'imp_4']
    return keys

def get_usable_keys():
    keys = ['acc', 'breath_1', 'breath_2', 'ecg', 'temp_1', 'temp_2']
    return keys

def get_artifact_keys():
    keys = ['acc', 'breath_1', 'breath_2', 'ecg', 'temp_1', 'temp_2']
    return keys

def get_result_keys():
    keys = ['n_steps', 'mean_activity', 'bpm', 'hrv', 'pnn50', 
            'rpm_tho', 'rpm_abd', 'amp_tho', 'amp_abd', 'amp_ecg',
            'temp_right', 'temp_left']
    return keys

def get_sig_names():
    
    sig_names = {'acc':                     'Accel', 
                 'accx':                    'Acc x',
                 'accy':                    'Acc y',
                 'accz':                    'Acc z',
                 'breath':                  'Breath',
                 'breath_1':                '$Resp_{Tho}$',
                 'breath_2':                '$Resp_{Abd}$',
                 'ecg':                     'ECG', 
                 'temp':                    'TEMP', 
                 'temp_1':                  '$T°C_{right}$', 
                 'temp_2':                  '$T°C_{left}$', 
                 'temp_right':              'Right T°C',
                 'temp_left':               'Left T°C',
                 'imp':                     'IMP',
                 'imp_1':                   'Imp 1',
                 'imp_2':                   'Imp 2',
                 'imp_3':                   'Imp 3',
                 'imp_4':                   'Imp 4',
                 'ecg_breath':              'ECG + Breath',
                 'ecg_temp':                'ECG + Temp',
                 'ecg_breath_temp':         'ALL', 
                 'n_steps':                 'N steps',
                 'steps':                   'N steps',
                 'mean_activity':           'Level of Intensity',
                 'mean_activity_level':     'Mean activity level',
                 'bpm':                     'Heart rate (BPM)',
                 'hrv':                     'HRV (ms)',
                 'pnn50':                   'pNN50 (%)',
                 'amp_tho':                 'THO peaks Amplitude (mV)',
                 'amp_abd':                 'ABD peaks Amplitude (mV)',
                 'amp_ecg':                 'ECG peaks Amplitude (mV)',
                 'rpm_tho':                 'Respiratory rate (thoracic)',
                 'rpm_abd':                 'Respiratory rate (abdominal)',
                 'rpm_1':                   'RPM (Thoracic)',
                 'rpm_2':                   'RPM (Abdominal)',
                 }
    return sig_names

def get_sig_ranges():
    
    sig_names = {'bpm':                 (40, 160), 
                 'hrv':                 (-5, 800),
                 'pnn50':               (-5, 105),
                 'n_steps':             (-5, 300),
                 'mean_activity_level': (-5, 100),
                 'mean_activity':       (-5, 100),
                 'temp_right':          (20, 45),
                 'temp_left':           (20, 45),
                 'temp_1':          (   20, 45),
                 'temp_2':              (20, 45),
                 'rpm_tho':             (5, 50),
                 'rpm_abd':             (5, 50),
                 'amp_tho':             (-0.1, 2.4),
                 'amp_abd':             (-0.1, 2.4),
                 'rpm_1':               (5, 50),
                 'rpm_2':               (5, 50),
                 'amp_1':               (-0.1, 2.4),
                 'amp_2':               (-0.1, 2.4),
                 'amp_ecg':             (-0.1, 2.4),
                 'percentage':          (-5, 105),
                 }
    return sig_names
    
def show_global_info_from_report(report, fontsize=16, path_save=None):
    
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    if len(report['disconnection']) == 0:
        return
    # Disconnection
    plt.figure()
    data = report['disconnection']['percentage']
    data = data[data != 'None'].astype('float')
    sig_types_wanted = get_summary_keys()
    sig_types = np.unique(data.index)
   
    sig_colors = get_sig_colors()
    sig_names = get_sig_names()
    values = []
    cols_graph = []
    colors = []
    for sig_type in sig_types:
        if sig_type in sig_types_wanted:
            if not np.isnan(data.loc[sig_type]):
                if sig_type == 'accx':
                    cols_graph.append('ACC')
                else:
                    cols_graph.append(sig_names[sig_type])
                colors.append(sig_colors[sig_type][1])
                values.append(data.loc[sig_type])
    plt.bar(cols_graph, values, color=colors, zorder=3)
    plt.grid(zorder=0, axis='y')
    plt.xticks(fontsize=fontsize)
    plt.ylim(0, 105)
    plt.title('Disconnection (%)', fontsize=fontsize)
    plt.ylabel('%', fontsize=fontsize)
    for i, val in enumerate(values):
        if np.isnan(val):
            val = 0
        plt.text(cols_graph[i], val + 1, str(int(val)), fontsize=fontsize)
        
    if path_save is not None:
        plt.savefig(path_save + 'global_disconnection_bar.png')
    
    # Usable signal
    plt.figure()
    data = report['usable']['percentage']
    data = data[data != 'None'].astype('float')
    sig_types_wanted = get_usable_keys()
    sig_types = np.unique(data.index)
    values = []
    cols_graph = []
    colors = []
    for sig_type in sig_types:
        if sig_type in sig_types_wanted:
            if not np.isnan(data.loc[sig_type]):
                cols_graph.append(sig_names[sig_type])
                colors.append(sig_colors[sig_type][1])
                values.append(data.loc[sig_type])
                
    plt.bar(cols_graph, values, color=colors, zorder=3)
    plt.grid(zorder=0, axis='y')
    plt.xticks(fontsize=fontsize)
    plt.ylim(0, 105)
    plt.title('Usable signal (%)', fontsize=fontsize)
    plt.ylabel('%', fontsize=fontsize)
    for i, val in enumerate(values):
        if np.isnan(val):
            val = 0
        plt.text(cols_graph[i], val + 1, str(int(val)), fontsize=fontsize)
    
    if path_save is not None:
        plt.savefig(path_save + 'global_usable_bar.png')
        
    # # artifact
    # plt.figure()
    # data = report['artifact']['percentage']
    # data = data[data != 'None'].astype('float')
    # sig_types_wanted = get_artifact_keys()
    # sig_types = np.unique(data.index)
    # values = []
    # cols_graph = []
    # colors = []
    # for sig_type in sig_types:
    #     if sig_type in sig_types_wanted:
    #         cols_graph.append(sig_names[sig_type])
    #         colors.append(sig_colors[sig_type][0])
    #         values.append(data.loc[sig_type])
    
    # plt.bar(cols_graph, values, color=colors, zorder=3)
    # plt.grid(zorder=0, axis='y')
    # plt.xticks(fontsize=fontsize)
    # plt.ylim(0, 105)
    # plt.title('artifact (%)', fontsize=fontsize)
    # plt.ylabel('%', fontsize=fontsize)
    # for i, val in enumerate(values):
    #     if np.isnan(val):
    #         val = 0
    #     plt.text(cols_graph[i], val + 1, str(int(val)), fontsize=fontsize)
    
    # if path_save is not None:
    #     plt.savefig(path_save + 'global_artifact_bar.png')
        
def show_plot_from_report_time_split(df, row, col, hue, keys, label, fontsize=18, path_save=None):

    for key in df.index.unique():
        if key in keys:
            if col == 'result':
                ylim = get_sig_ranges()[key]
            else:
                ylim = get_sig_ranges()[col]
            df_key = df[df.index == key]
            plt.figure()
            
            show_result_plot(df=df_key, row=row, col=col, key=key, ylim=ylim, 
                             fontsize=fontsize, yoffset=5, ytype='int', hue=hue, 
                             flag_legend=True, disp_minmax=True)
            plt.title((get_sig_names()[key] + ' ' + label), fontsize=fontsize)
            
            if path_save is not None:
                plt.savefig(path_save + label + '_' + key + '_plot.png')
            
def show_bar_from_report_time_split(df, row, col, hue, keys, label, df_mean_acc,\
                                    fontsize=18, path_save=None):
    
    # idx = np.where(df[col].isnull() == False)
    # df = df.iloc[idx]
    for key in df.index.unique():
        if key in keys:
            df_key = df[df.index == key]
            if col == 'result':
                ylim = get_sig_ranges()[key]
            else:
                ylim = get_sig_ranges()[col]
            plt.figure()
            show_result_bar(df=df_key, row=row, col=col, key=key, ylim=ylim, 
                             fontsize=fontsize, 
                             yoffset=5, ytype='int', hue=hue,
                             flag_legend=True, disp_minmax=False)
            plt.title((get_sig_names()[key] + ' ' + label), fontsize=fontsize)
            if path_save is not None:
                plt.savefig(path_save + label + '_' + key + '_bar.png')

def show_subbar_from_report_time_split(df, row, col, hue, keys, label, 
                                    fontsize=18, path_save=None):
    
    plt.figure(figsize=(12,8))
    for key in df.index.unique():
        if key in keys:
            df_key = df[df.index == key]
            if col == 'result':
                if key == 'bpm':
                    plt.subplot(4, 1, 1)
                elif key == 'rpm_abd':
                    plt.subplot(4, 1, 2)
                elif key == 'temp_right':
                    plt.subplot(4, 1, 3)
                elif key == 'mean_activity':
                    plt.subplot(4, 1, 4)
                else:
                    continue
                ylim = get_sig_ranges()[key]
                show_result_bar(df=df_key, row=row, col=col, key=key, ylim=ylim, 
                                fontsize=fontsize, 
                                yoffset=5, ytype='int', hue=hue,
                                flag_legend=True, disp_minmax=False)
                plt.title('')
                plt.yticks(fontsize=12)
                plt.text(0, ylim[1]*.8, get_sig_names()[key], fontsize=15)
                if key != 'mean_activity':
                    plt.xlabel('')
                    plt.xticks([])
            
            if path_save is not None:
                plt.savefig(path_save + 'results_bar.png')
                
def show_dist_from_report_time_split(df, col, hue, keys, label, fontsize=18, path_save=None):
    for key in df.index.unique():
        if key in keys:
            df_key = df[df.index == key]
            if col == 'result':
                xlim = get_sig_ranges()[key]
            else:
                xlim = get_sig_ranges()[col]
            plt.figure()
            show_result_dist(df=df_key, col=col, key=key, hue=hue, xlim=xlim, fontsize=fontsize)
            plt.title((get_sig_names()[key] + ' ' + label), fontsize=fontsize)
            plt.xlabel('')
            if path_save is not None:
                plt.savefig(path_save + label + '_' + key + '_dist.png')

def savefig_from_report(report, fontsize=12, path_save=None):
    sns.set_style('dark')
    show_global_info_from_report(report, fontsize=fontsize, path_save=path_save)
    plt.close('all')
        
def savefig_from_report_time_split(report_time_split, fontsize=12, path_save=None):
    usable = report_time_split['usable']
    # artifact = report_time_split['artifact']
    result = report_time_split['result']
    activity_threshold = 10
    df_mean_acc = result[result.index == 'mean_activity']
  
    hue = df_mean_acc.result.astype('float').values >= activity_threshold

    sns.set_style('dark')
    plt.close('all')
    keys = get_usable_keys()
    # show_plot_from_report_time_split(df=usable, row='start', col='percentage', 
    #                                   hue=hue, keys=keys, label='usable', fontsize=fontsize, 
    #                                   path_save=path_save)
    plt.close('all')
    show_bar_from_report_time_split(df=usable, row='start', col='percentage', 
                                    hue=hue, keys=keys, label='usable', 
                                    df_mean_acc=df_mean_acc, fontsize=fontsize, 
                                    path_save=path_save)
    plt.close('all')
    show_dist_from_report_time_split(df=usable, col='percentage', hue=None, keys=keys,
                                     label='usable', fontsize=fontsize, 
                                     path_save=path_save)
    plt.close('all')
    # keys = get_artifact_keys()
    # show_plot_from_report_time_split(df=artifact, row='start', col='percentage', 
    #                                   hue=hue, keys=keys, label='artifact', fontsize=fontsize, 
    #                                   path_save=path_save)
    # plt.close('all')
    # show_bar_from_report_time_split(df=artifact, row='start', col='percentage', 
    #                                 hue=hue, keys=keys, label='artifact', fontsize=fontsize, 
    #                                 path_save=path_save)
    # plt.close('all')
    # show_dist_from_report_time_split(df=artifact, col='percentage', hue=hue, keys=keys,
    #                                   label='artifact', fontsize=fontsize, 
    #                                   path_save=path_save)
    # plt.close('all')
    
    keys = get_result_keys()
    # show_plot_from_report_time_split(df=result, row='start', col='result', 
    #                                   hue=hue, keys=keys, label='result', fontsize=fontsize, 
    #                                   path_save=path_save)
    plt.close('all')
    show_bar_from_report_time_split(df=result, row='start', col='result', 
                                      hue=hue, keys=keys, label='result', 
                                      df_mean_acc=df_mean_acc, fontsize=fontsize, 
                                      path_save=path_save)
    plt.close('all')
    show_subbar_from_report_time_split(df=result, row='start', col='result', 
                                      hue=hue, keys=keys, label='result', 
                                      fontsize=12, 
                                      path_save=path_save)
    plt.close('all')
    show_dist_from_report_time_split(df=result, col='result', hue=None, keys=keys,
                                      label='result', fontsize=fontsize, 
                                      path_save=path_save)
    plt.close('all')
