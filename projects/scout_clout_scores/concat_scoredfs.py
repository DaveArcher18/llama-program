import os
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt



def make_zeros(n):
    zero_list = []
    for i in range(n):
        zero_list.append(0)
    return zero_list    

def make_dfs(directory, foldername): 
    """
    Takes a directory containing csvs (directory) and a name for an output folder and places 2 csvs in the output folder.
    First the function concatenates all the dfs recording the users' scores over time. The second csv contains some summary
    statistics regarding each user.
    """
    filenames = os.listdir(directory)
    user_scores_over_time = {}

    for j in range(len(filenames)):
        #reads in each csv
        spreadsheet = pd.read_csv(os.path.join(directory, filenames[j]))

        for i in range(len(spreadsheet)):
            #checks if the user has appeared before - if not it appends zeros to their score vector
            if spreadsheet['username'][i] not in list(user_scores_over_time.keys()):
                user_scores_over_time[spreadsheet['username'][i]] = make_zeros(j)
            #appends the user's current score to their score vector
            user_scores_over_time[spreadsheet['username'][i]].append(spreadsheet['score'][i])
        
        for k in user_scores_over_time:
            #checks if the user has a score vector but isn't in the current df, if this is the case
            #their last score is added to their score vector
            if k not in list(spreadsheet['username']):
                user_scores_over_time[k].append(user_scores_over_time[k][-1])


    scores_df = pd.DataFrame.from_dict(user_scores_over_time)

    #makes a folder
    if foldername not in os.listdir('.'):
        os.mkdir(foldername)


    scores_df.to_csv(os.path.join(foldername, 'user_scores_spreadsheet.csv'))


    data = [[scores_df.columns[0], round(np.mean(np.array(scores_df[scores_df.columns[0]])), 2), round(np.std(np.array(scores_df[scores_df.columns[0]])), 2), list(scores_df[scores_df.columns[0]])[0] - list(scores_df[scores_df.columns[0]])[-1]]]
    for i in range(1, len(scores_df.columns)):
        data.append([scores_df.columns[i], round(np.mean(np.array(scores_df[scores_df.columns[i]])), 2), round(np.std(np.array(scores_df[scores_df.columns[i]])), 2), list(scores_df[scores_df.columns[i]])[0] - list(scores_df[scores_df.columns[i]])[-1]])

    summary_statistics_df = pd.DataFrame(data, columns = ['username', 'mean_score', 'std_score' ,'score_change'])
    summary_statistics_df.to_csv(os.path.join(foldername, 'summary_statistics_spreadsheet.csv'))

make_dfs('scout_signups_clean', 'ScoutSummaries')

make_dfs('wtw_signups_clean', 'WTWSummaries')








### Code For Plotting 
'''
fig, ax = plt.subplots(figsize = (12, 12))


for i in user_scores_over_time:
        ax.plot(user_scores_over_time[i])


plt.show()
'''
