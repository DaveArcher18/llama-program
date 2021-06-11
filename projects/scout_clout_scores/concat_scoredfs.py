import os
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def make_scores_df(directory): 
    """
    Takes a directory containing csvs (directory) The function concatenates all the dfs recording the users' scores over time. 
    """
    csv_list = []
    filenames = os.listdir(directory)
    user_scores_over_time = {}

    for j in range(len(filenames)):
        #reads in each csv
        cur_df = pd.read_csv(os.path.join(directory,filenames[j]))
        cur_df = cur_df[['username', 'score']]
        cur_df = pd.pivot_table(cur_df, values='score',
                            columns=['username'])
        
        cur_df['timestamp'] = datetime.datetime.fromtimestamp(int(filenames[j].split('_')[-1].split('.')[0]))
        
        cur_df.set_index('timestamp', inplace = True)
        csv_list.append(cur_df)

    df_merged = pd.concat(csv_list)
    df_merged = df_merged.sort_values('timestamp')    
    
    return df_merged



def make_csv(df, foldername, filename, timestamp):   
    if foldername not in os.listdir('.'):
        os.mkdir(foldername)

    df.to_csv(os.path.join(foldername, filename + " _" + str(int(timestamp)) + '.csv'))


def find_perc_diffs(df_column, previous_value):
    perc_change = round(((df_column[-1] - previous_value)/previous_value)*100,2)

    return perc_change

def make_summary(df):

    perc_change_array = [] 

    for i in df.columns:     
        perc_change_array.append(find_perc_diffs(df[i].dropna(), df[i].dropna()[0]))
    
    summary = df.describe()

    summary = summary.append(pd.DataFrame([perc_change_array], columns=summary.columns, index = ['Percentage Change']), ignore_index=False)

    return summary.round(3)

scout = make_scores_df('scout_signups_clean')

summary_scout = make_summary(scout)

make_csv(summary_scout , 'ScoutSummaries', 'Scout_summary', datetime.datetime.timestamp(scout.index[-1]))


wtw = make_scores_df('wtw_signups_clean')

summary_wtw = make_summary(wtw)

make_csv(summary_wtw, 'WTWSummaries', "WTW_summary", datetime.datetime.timestamp(wtw.index[-1]))


'''
#to show that the timestamp stuff works
x = scout.index[-1]

print(x) 

x = int(datetime.datetime.timestamp(x))

print(x)

x = datetime.datetime.fromtimestamp(x)

print(x)
'''