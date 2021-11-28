import os
import sys
import numpy as np

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from functions import count_t_student, count_t_student_in_excel, generate_boxplots, load_data, get_each_time, compare_data, filter_ICA, plot_evoked_data, create_file, create_file_writer, save_to_file
files_no = ('01', '02', '03', '04', '05', '08', '09', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20', '21', '23', '24', '25',
'26', '28', '30')
times = ('first_ten_minutes', 'last_ten_minutes')
save=True
_save_to_file = True
display_charts = False
find_max = True 
auto_reject = True

if save is True:
        file = create_file()
        file_writer = create_file_writer(file)

for number in files_no: 
    data = load_data(file_no=number) 
    first_ten_minutes_evoked = None
    last_ten_minutes_evoked = None
    data_after_ica = filter_ICA(data, save, display_charts) 
    for time in times:
        each_data, length_in_seconds = get_each_time(data_after_ica, time)
        if time == 'first_ten_minutes':
            first_ten_minutes_evokeds = plot_evoked_data(each_data, save, auto_reject=auto_reject, channels_to_pick=["Cz", "Pz"])
        else:
            last_ten_minutes_evokeds = plot_evoked_data(each_data, save, auto_reject=auto_reject, channels_to_pick=["Cz", "Pz"])
        del each_data
    evokeds_array = np.stack((first_ten_minutes_evokeds,last_ten_minutes_evokeds),axis=1)
    for evoked in evokeds_array:
        compare_data(evoked, save=save, find_max=True, file_writer=file_writer, display_charts=display_charts, _save_to_file=_save_to_file, length = length_in_seconds)
file.close()
count_t_student_in_excel()

generate_boxplots()
del last_ten_minutes_evokeds, first_ten_minutes_evokeds 
