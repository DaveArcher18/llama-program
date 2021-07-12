import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import os

paycity_1_clicks = pd.read_csv('paycity_1_clicks.csv')
paycity_1_sessions = pd.read_csv('paycity_1_sessions.csv')

corium_1_clicks = pd.read_csv('corium_1_clicks.csv')
corium_1_sessions = pd.read_csv('corium_1_sessions.csv')


def plot_total_over_time(data, title, out_dir):
    
    '''Plots total Clicks/Sessions over time for a campaign
    data is a pd.DataFrame
    title is the graph title
    out_dir is the name of the file where the graph will be stored'''

    times = data['time_stamp']
    values = np.array(data.sum(axis=1))
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.plot(times, values)
    
    ax.set_title(title)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{title}.png')

def plot_for_each_user(data, title, out_dir):
    
    '''Plots the clicks/sessions over time for each user involved in a campaign
    data is a pd.DataFrame
    title is the graph title
    out_dir is the name of the file where the graph will be stored'''
    
    times = data['time_stamp']
    cols = list(data.columns)[1:]
    fig, ax = plt.subplots(figsize = (20, 20))
    for i in cols:
        ax.plot(times,  np.array(data[i]), label = i)

    ax.legend()
    ax.set_title(title)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{title}.png')


def plot_sessions_and_clicks_over_time(clicks_data, sessions_data, title, out_dir):
    
    '''Plots total Clicks and Sessions over time for a campaign
    data is a pd.DataFrame
    title is the graph title
    out_dir is the name of the file where the graph will be stored'''

    times = clicks_data['time_stamp']
    clicks_values = np.array(clicks_data.sum(axis=1))
    sessions_values = np.array(sessions_data.sum(axis=1))

    
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.plot(times,  clicks_values, label = 'Clicks')
    ax.plot(times,  sessions_values, label = "Sessions")
    
    ax.legend()
    ax.set_title(title)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{title}.png')




plot_sessions_and_clicks_over_time(paycity_1_clicks, paycity_1_sessions, 'Plot showing clicks and sessions over time for paycity_1', 'paycity_1_graphs')