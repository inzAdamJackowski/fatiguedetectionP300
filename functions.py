from os import path, stat
import shutil
import matplotlib
from mne.io.pick import channel_type, pick_channels_evoked
from mne.io.proj import Projection
from mne.utils import dataframe
from numpy.core.numeric import True_
from numpy.core.shape_base import block
import pandas as pd
from pandas.core.frame import DataFrame
import scipy.io
import os
import numpy as np
import mne
import time
import csv
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap, compute_proj_eog)
from autoreject import (AutoReject, set_matplotlib_defaults, Ransac)
import os
import sys
from scipy.stats import ttest_ind
from varname import nameof
import matplotlib.pyplot as plt


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

def save_figure(figure, number, peroid_type, title):
    if peroid_type !=None:
        path_to_figure = os.path.join("E:\Mgr\wyniki", str(number), str(peroid_type))
    else:
        path_to_figure = os.path.join("E:\Mgr\wyniki", str(number))
    if not path.exists(path_to_figure): 
        os.makedirs(path_to_figure)
    figure.savefig(os.path.join(path_to_figure, str(title)))
    matplotlib.pyplot.close('all')
    del figure

def convert_to_epoch(raw, ref=[], tmin=-0.5, tmax=2.0):
    events = mne.find_events(raw, stim_channel='Event')
    reject_criteria = dict(eeg=200e-6)
    epochs_params = dict(events=events, event_id=1, tmin=tmin, tmax=tmax, reject=reject_criteria, preload=True, baseline=(-0.1, 0.0))

    raw, _ = mne.set_eeg_reference(inst=raw, ref_channels=ref, ch_type='eeg', projection = False)
    picks = mne.pick_types(raw.info, eeg=True)
    raw_epoch = mne.Epochs(raw, **epochs_params)
    #raw_epoch.drop_bad()
    return raw_epoch

def reject_bad_data(epoch):
    ar = AutoReject()
    try:
        epochs_clean = ar.fit_transform(epoch)
    except:
        print("CAN'T REPAIR EPOCHS")
        epochs_clean = epoch
    ransac = Ransac()
    epochs_clean_interpolated = ransac.fit_transform(epochs_clean)
    print("-------------BAD CHANNELS-----------------")
    print('\n'.join(ransac.bad_chs_))
    return epochs_clean_interpolated

def display_averaged_erps(epochs_clean, save=False, channels_to_pick="all", display_charts=False):
    # dodać parametr z listą kanałów lub domyślnie wszystkie
    evokeds_clean = list()
    if channels_to_pick != "all":
        for channel in channels_to_pick:
            epochs_clean_single_channel = epochs_clean.copy().pick_channels([channel])
            evoked_clean = epochs_clean_single_channel.average()
            evoked_clean.info['device_info'] = channel
            evokeds_clean.append(evoked_clean)

    evoked_clean = epochs_clean.average()
    title = 'EEG Original reference'

    if save is True:
        save_figure(evoked_clean.plot_topomap(times=[0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31], size=9., title=title, time_unit='s', show=False), evoked_clean.info['file_id'], evoked_clean.info['description'], "topo")
        save_figure(evoked_clean.plot(titles=dict(eeg=title), time_unit='s', show=False), evoked_clean.info['file_id'], evoked_clean.info['description'], "erps") 
    elif display_charts is True:
        evoked_clean.plot_topomap(times=[0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31], size=9., title=title, time_unit='s', show=True)
        evoked_clean.plot(titles=dict(eeg=title), time_unit='s', show=True)

    del evoked_clean, epochs_clean, epochs_clean_single_channel  # free up memory
    return np.array(evokeds_clean)


def filter_ICA(raw, save, display_charts):
    events = mne.find_events(raw, stim_channel='Event')
    #raw.plot(events=events, duration=10, block=True, n_channels=16, scalings="auto", show=True)
    raw = raw.load_data().filter(l_freq=1.0, h_freq=40.0)
    #raw.plot(events=events, duration=10, block=True, n_channels=16, scalings="auto", show=True)
    #Get events from an raw
    
    if save is True:
        save_figure(raw.plot(events=events, duration=10, block=True, n_channels=16, scalings="auto", show=False), raw.info['file_id'], "check_if_ica_works", "channels_before_ica"+ str(raw.info['file_id']))
    elif display_charts is True:
        raw.plot(events=events, duration=10, block=True, n_channels=16, scalings="auto", show=True)
    ica = ICA(n_components=16, random_state=97)
    ica.fit(raw)
    if save is True:
        raw.load_data()
        save_figure(ica.plot_sources(raw, show_scrollbars=False, show=False), raw.info['file_id'], "check_if_ica_works", "ica_sources" + str(raw.info['file_id']))
        save_figure(ica.plot_components(show=False)[0], raw.info['file_id'], "check_if_ica_works", "ica_components" + str(raw.info['file_id']))
    eog_indices_Fp1, eog_scores = ica.find_bads_eog(raw, ch_name = "Fp1")
    eog_indices_F3, eog_scores = ica.find_bads_eog(raw, ch_name = "F3")
    eog_indices_F4, eog_scores = ica.find_bads_eog(raw, ch_name = "F4")
    eog_indices_F7, eog_scores = ica.find_bads_eog(raw, ch_name = "F7")
    eog_indices_F8, eog_scores = ica.find_bads_eog(raw, ch_name = "F8")
    eog_indices_F9, eog_scores = ica.find_bads_eog(raw, ch_name = "F9")
    eog_indices_Fz, eog_scores = ica.find_bads_eog(raw, ch_name = "Fz")
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name="Fz")
    sources_to_remove = eog_indices_Fp1 + eog_indices_F4 +  eog_indices_F8 + eog_indices_F9 + eog_indices_Fz + ecg_indices + eog_indices_F3 + eog_indices_F7# + eog_indices_F3 + ecg_indices
    print("--------------------ICA SOURCES TO REMOVE--------------------")
    print(sources_to_remove)
    if len(sources_to_remove) == 0:
        eog_indices_Fp1 = []
        ecg_indices = []
        print("--------------------ICA SOURCES TO REMOVE MANUALY--------------------")
        print(eog_indices_Fp1 + ecg_indices)
        print("--------------------EOG--------------------")
        print(eog_indices_Fp1)
        #print("--------------------ECG--------------------")
        #print(ecg_indices)
    ica.exclude = sources_to_remove
    raw = ica.apply(raw)
    
    if save is True:
        save_figure(raw.plot(events=events, duration=10, block=True, n_channels=16, scalings="auto", show=False), raw.info['file_id'], "check_if_ica_works", "channels_after_ica" + str(raw.info['file_id'] +"_removed" + str(sources_to_remove)))
    elif display_charts is True:
        raw.plot(events=events, duration=10, block=True, n_channels=16, scalings="auto", show=True)
    return raw

def filter_SSP(raw):
    #Get events from an raw
    events = mne.find_events(raw, stim_channel='Event')
    raw.plot(events=events, duration=5, block=True, n_channels=16, scalings="auto", show=False)
    eog_projs, _ = compute_proj_eog(raw, n_eeg=1, reject=None,
                                no_proj=True, ch_name = "Fp1", average=True)
    raw.del_proj().set_eeg_reference(projection=True).apply_proj()
    raw.add_proj(eog_projs)
    raw.plot(events=events, duration=10, block=True, n_channels=16, scalings="auto", show=False)
    return raw

def load_data(file_no='26'):
    # Load data
    mat = scipy.io.loadmat('E:\Mgr\dane\S' + str(file_no) + '_EEG.bin.mat')
    first_array = mat['eeg']
    result_array = list()
    for arrays in first_array:
        result_array.append((arrays[:17]))
    #---------------------------------------------------------------------------
    #Create numpy array from array and transpose it
    data = np.array(result_array)
    data_t = np.transpose(data)
    events = data_t[16]
    data_t = data_t[:16] / 1000000  
    data_t = np.append(data_t, [events], axis=0)                                                  

    #Create raw objects from array
    ch_names = ["Oz", "O2", "O1", "Pz", "P4", "P3", "C4", "C3",
        "Cz", "F8", "F7", "Fz", "F4", "F3", "Fp1", "F9", "Event"]
    info = mne.create_info(ch_names, 512, 'eeg')
    #info['bads'].extend(['Fp1', 'F9'])
    info['file_id'] = str(file_no)
    # add a list of channels
    raw = mne.io.RawArray(data_t, info)
    #raw.info["experimenter"] = raw._last_time
    raw.set_channel_types(mapping={'Event': 'stim'}) 
    raw.set_montage('standard_1020')
    #a = mne.channels.make_standard_montage('standard_1020')
    #a.plot(show_names=False)
    #raw.plot_sensors(ch_type='eeg', show_names=True, block=True)
    create_dir(file_no)
    del mat
    return raw

def create_dir(file_no):
    dir_path = os.path.join('E:\Mgr\wyniki', str(file_no))
    if path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, mode=777)

def create_file():
    file = open('local_max.csv', 'w', newline='')
    return file

def create_file_writer(file):
    fnames = ['file number', "channel name", "time of local max f10 (s)",
     "value of local max f10 (uV)", "time of local max l10 (s)",
      "value of local max l10 (uV)", "is last 10 minutes lower", "is last 10 minutes later", "length (s)"]
    writer = csv.DictWriter(file, fieldnames=fnames)
    writer.writeheader()
    return writer


def get_each_time(raw, time):
    #Divide array into first and last 10 minutes
    final_raw = None
    length_in_seconds = raw._last_time
    if time == 'first_ten_minutes':
        final_raw = raw.copy().crop(tmin=0.0, tmax=600.0)
    elif time == 'last_ten_minutes':
        final_raw = raw.copy().crop(tmin=float(raw._last_time-600.0), tmax=float(raw._last_time))
    final_raw.info['description'] = time 
    return final_raw, length_in_seconds


def plot_evoked_data(raw, save, auto_reject=True, channels_to_pick="all", display_charts=False):
    epoch = convert_to_epoch(raw, ref=[])
    if auto_reject is True:
        repaired_epoch = reject_bad_data(epoch)
    else:
        repaired_epoch = epoch
    return display_averaged_erps(repaired_epoch, save=save, channels_to_pick=channels_to_pick, display_charts=display_charts)

def compare_data(evoked, save, find_max=False, file_writer=None, display_charts=False, _save_to_file=True, length=None):
    # get each time
    # find max
    # compare
    if find_max is True:
        first_ten_minutes_max, last_ten_minutes_max = compare_max(evoked, file_writer, save, display_charts=display_charts, _save_to_file=True, length=length)

    if save is True and find_max is True:
        figure= mne.viz.plot_compare_evokeds(dict(first_ten_minutes=evoked[0], last_ten_minutes=evoked[1]),
                             legend='lower right', show_sensors='upper right', title=str(evoked[0].info['device_info'] + evoked[0].info['file_id']), show=False)[0]
        figure.axes[0].annotate('local max first ten minutes', xy=(first_ten_minutes_max[0], first_ten_minutes_max[1]), xytext=(1, 6), arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
        figure.axes[0].annotate('local max last ten minutes', xy=(last_ten_minutes_max[0], last_ten_minutes_max[1]), xytext=(1, 5), arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
        figure.axes[0].scatter(first_ten_minutes_max[0], first_ten_minutes_max[1])
        figure.axes[0].scatter(last_ten_minutes_max[0], last_ten_minutes_max[1])
        figure.axes[0].set_ylim([None,9])
        
        save_figure(figure, evoked[0].info['file_id'], "compare", str("compare_" + evoked[0].ch_names[0] + "_" + evoked[0].info['file_id'])) 
    
    elif save is True and find_max is False:

        save_figure(mne.viz.plot_compare_evokeds(dict(first_ten_minutes=evoked[0], last_ten_minutes=evoked[1]),
                             legend='lower right', show_sensors='upper right', title=str(evoked[0].info['device_info'] + evoked[0].info['file_id']) , show=False)[0], evoked[0].info['file_id'], "compare", str("compare_" + evoked[0].ch_names[0] + "_" + evoked[0].info['file_id'])) 
    elif display_charts is True:
        mne.viz.plot_compare_evokeds(dict(first_ten_minutes=evoked[0], last_ten_minutes=evoked[1]),
                             legend='lower right', show_sensors='lower center', title=str(evoked[0].info['device_info'] + evoked[0].info['file_id']))

def compare_max(evokeds_to_compare, writer, save, display_charts=False, _save_to_file=True, length=None):
    evoked_first_ten_minutes = None
    evoked_last_ten_minutes = None
    local_max_raw_values_first_ten_minutes = None
    local_max_raw_values_last_ten_minutes = None
    for evoked in evokeds_to_compare:
        channel_name = evoked.ch_names[0]
        cropped_evoked = evoked.copy().crop(tmin=0.250, tmax=0.600)
        if cropped_evoked.info['description'] == "first_ten_minutes":
            evoked_first_ten_minutes = cropped_evoked
        else:
            evoked_last_ten_minutes = cropped_evoked
        frame_cropped_evoked = cropped_evoked.to_data_frame()
        local_max = np.max(frame_cropped_evoked[channel_name])
        local_max_raw = frame_cropped_evoked.loc[frame_cropped_evoked[channel_name] == local_max]
        print(local_max_raw)
        if cropped_evoked.info['description'] == "first_ten_minutes":
            local_max_raw_values_first_ten_minutes = [local_max_raw.iloc[0]['time']/1000.0, local_max_raw.iloc[0][channel_name]] 
        else:
            local_max_raw_values_last_ten_minutes = [local_max_raw.iloc[0]['time']/1000.0, local_max_raw.iloc[0][channel_name]] 
    is_last_ten_minutes_lower = float(local_max_raw_values_first_ten_minutes[1]) > float(local_max_raw_values_last_ten_minutes[1]) 
    is_last_ten_minutes_later = float(local_max_raw_values_first_ten_minutes[0]) < float(local_max_raw_values_last_ten_minutes[0])
    if save is True:
        save_figure(mne.viz.plot_compare_evokeds(dict(first_ten_minutes=evoked_first_ten_minutes, last_ten_minutes=evoked_last_ten_minutes),
                             legend='lower right', show_sensors='lower center', title=str(evoked_first_ten_minutes.info['device_info'] + evoked_first_ten_minutes.info['file_id']) , show=False)[0], evoked_first_ten_minutes.info['file_id'], "compare", str("compare_local_max_" + evoked_first_ten_minutes.ch_names[0] + "_" + evoked_first_ten_minutes.info['file_id'])) 
    if _save_to_file is True:
        save_to_file(writer, cropped_evoked.info['file_id'], cropped_evoked.ch_names[0], local_max_raw_values_first_ten_minutes[0],
            local_max_raw_values_first_ten_minutes[1], local_max_raw_values_last_ten_minutes[0], 
            local_max_raw_values_last_ten_minutes[1], is_last_ten_minutes_lower, is_last_ten_minutes_later, length) 
    if display_charts is True:    
        mne.viz.plot_compare_evokeds(dict(first_ten_minutes=evoked_first_ten_minutes, last_ten_minutes=evoked_last_ten_minutes),
        legend='lower right', show_sensors='lower center')
    return local_max_raw_values_first_ten_minutes, local_max_raw_values_last_ten_minutes
    
def save_to_file(file_writer, file_number, channel_name, time_of_local_max_first_ten_minutes,
 value_of_local_max_first_ten_minutes, time_of_local_max_last_ten_minutes,
  value_of_local_max_last_ten_minutes, is_last_ten_minutes_lower, is_last_ten_minutes_later, length):
    file_writer.writerow({'file number' : str(file_number),
     'channel name': str(channel_name),
     'time of local max f10 (s)': str(time_of_local_max_first_ten_minutes),
     'value of local max f10 (uV)': str(value_of_local_max_first_ten_minutes),
      'time of local max l10 (s)': str(time_of_local_max_last_ten_minutes),
      'value of local max l10 (uV)': str(value_of_local_max_last_ten_minutes),
      'is last 10 minutes lower': bool(is_last_ten_minutes_lower),
      'is last 10 minutes later': bool(is_last_ten_minutes_later),
      'length (s)': str(length)})

def export_csv_to_excel(csv_path = file_dir + '\local_max.csv', excel_path = 'E:\Mgr\wyniki' + '\local_max.xlsx'):
    # export file as xlsx
    read_file = pd.read_csv(csv_path)
    read_file.to_excel(excel_path, index = False, sheet_name='local_max')

def count_t_student(data):
         #value
        mean_local_max_f10 = data["value of local max f10 (uV)"].mean()
        variance_local_max_f10 = data["value of local max f10 (uV)"].var()
        mean_local_max_l10 = data["value of local max l10 (uV)"].mean()
        variance_local_max_l10 = data["value of local max l10 (uV)"].var()
        t_student_v, p_v = ttest_ind(data["value of local max f10 (uV)"], data["value of local max l10 (uV)"])
        
        #time
        mean_time_of_local_max_f10 = data["time of local max f10 (s)"].mean()
        variance_time_of_local_max_f10 = data["time of local max f10 (s)"].var()
        mean_time_of_local_max_l10 = data["time of local max l10 (s)"].mean()
        variance_time_of_local_max_l10 = data["time of local max l10 (s)"].var()
        t_student_t, p_t = ttest_ind(data["time of local max f10 (s)"], data["time of local max l10 (s)"])
        
        degrees_of_freedom = 2 * len(data) -2
        return mean_local_max_f10, variance_local_max_f10, mean_local_max_l10, variance_local_max_l10, mean_time_of_local_max_f10, variance_time_of_local_max_f10, mean_time_of_local_max_l10, variance_time_of_local_max_l10, t_student_v, p_v, t_student_t, p_t, degrees_of_freedom

def filter_data_in_sheets():
    export_csv_to_excel()
    all_data = pd.read_excel('E:\Mgr\wyniki' + '\local_max.xlsx', sheet_name='local_max')
    Cz = all_data[(all_data["channel name"] == 'Cz')]
    Cz_only_true = all_data[np.logical_and((all_data["channel name"] == 'Cz'), (all_data["is last 10 minutes lower"] == True))]
    Cz_only_true.at[Cz_only_true.index[0],'channel name'] = 'Cz_only_true'
    Pz = all_data[(all_data["channel name"] == 'Pz')]
    Pz_only_true = all_data[np.logical_and((all_data["channel name"] == 'Pz'), (all_data["is last 10 minutes lower"] == True))]
    Pz_only_true.at[Pz_only_true.index[0],'channel name'] = 'Pz_only_true'
    #create new dataArray
    t_students = pd.DataFrame(columns=['sheet name', 'mean local max f10', 'variance local max f10',
        'mean local max l10', 'variance local max l10','mean time local max f10', 'variance time local max f10',
        'mean time local max l10', 'variance time local max l10', 't-student_v', 'p-value_v',
        't-student_t', 'p-value_t', 'degrees of freedom'])
    return Cz, Cz_only_true, Pz, Pz_only_true, t_students

def count_t_student_in_excel():
    Cz, Cz_only_true, Pz, Pz_only_true, t_students = filter_data_in_sheets()
    for channel in [Cz, Cz_only_true, Pz, Pz_only_true]:
        channel_name = channel["channel name"].iloc[0].replace("_", " ")
        mean_local_max_f10, variance_local_max_f10, mean_local_max_l10, variance_local_max_l10, mean_time_of_local_max_f10, variance_time_of_local_max_f10, mean_time_of_local_max_l10, variance_time_of_local_max_l10, t_student_v, p_v, t_student_t, p_t, degrees_of_freedom = count_t_student(channel)
       
        t_students = t_students.append({'sheet name': channel_name, "mean local max f10": mean_local_max_f10,
        "variance local max f10": variance_local_max_f10, "mean local max l10": mean_local_max_l10,
        "variance local max l10": variance_local_max_l10, "t-student_v": t_student_v, "p-value_v":p_v,
        "mean time local max f10": mean_time_of_local_max_f10, "variance time local max f10": variance_time_of_local_max_f10,
        "mean time local max l10": mean_time_of_local_max_l10, "variance time local max l10": variance_time_of_local_max_l10,
        "t-student_t": t_student_t, "p-value_t":p_t, "degrees of freedom": degrees_of_freedom}, ignore_index=True)
    
        with pd.ExcelWriter('E:\Mgr\wyniki' + '\local_max.xlsx', mode='a', engine='openpyxl') as writer: 
            channel.to_excel(writer, index = False, sheet_name=channel_name) 

    with pd.ExcelWriter('E:\Mgr\wyniki' + '\local_max.xlsx', mode='a', engine='openpyxl') as writer:
            t_students.to_excel(writer, index = False, sheet_name='T-student')
    
def generate_boxplots():
    excel = pd.read_excel('E:\Mgr\wyniki' + '\local_max.xlsx', sheet_name=["Cz", "Pz"])#["Cz only true", "Pz only true"])  
    for key, value in excel.items():

        boxplot = value.boxplot(column=['value of local max f10 (uV)', 'value of local max l10 (uV)'])
        boxplot.set_title('Compare values of local max of first and last 10 minutes')
        boxplot.set_ylabel('uV')
        save_figure(boxplot.figure, "boxplot", key.replace(" ", "_"), "local max values")

        boxplot = value.boxplot(column=['time of local max f10 (s)', 'time of local max l10 (s)'])
        boxplot.set_title('Compare times of local max of first and last 10 minutes')
        boxplot.set_ylabel('s')
        save_figure(boxplot.figure, "boxplot", key.replace(" ", "_"), "local max times")