from functions import save_figure
import pandas as pd

excel = pd.read_excel('E:\Mgr\porównanie' + '\local_max_11.xlsx', sheet_name=["Cz only true", "Pz only true"])  
for key, value in excel.items():

    boxplot = value.boxplot(column=['value of local max f10 (uV)', 'value of local max l10 (uV)'])
    boxplot.set_title('Compare values of local max of first and last 10 minutes')
    boxplot.set_ylabel('uV')
    save_figure(boxplot.figure, "local_max_11", key.replace(" ", "_"), "local max values")

    boxplot = value.boxplot(column=['time of local max f10 (s)', 'time of local max l10 (s)'])
    boxplot.set_title('Compare times of local max of first and last 10 minutes')
    boxplot.set_ylabel('s')
    save_figure(boxplot.figure, "local_max_11", key.replace(" ", "_"), "local max times")