import os 
import numpy as np
import pandas as pd 
import datetime 
import matplotlib.pyplot as plt
from fpdf import FPDF


########## Defining functions and classes ##########
def aggregate_data(data_list, filepath, column):
    '''Takes a list containing csvs, a filepath to the folder containing the csvs and a column name in the csv
    Returns a dataframe that has timestamp and slashtags as columns and the timestamps and column entires as rows.'''
    
    data0 = pd.read_csv(os.path.join(filepath, data_list[0]))
    timestamp0 = datetime.datetime.fromtimestamp(int(data_list[0].split('.')[0]))

    df = pd.DataFrame(columns = ['time_stamp'] + list(data0['slashtag']))

    df.loc[len(df.index)] = [timestamp0] + list(data0[column])

    for i in data_list[1:]:
        data = pd.read_csv(os.path.join(filepath, i))
        df.loc[len(df.index)] = [datetime.datetime.fromtimestamp(int(i.split('.')[0]))] + list(data[column])

    return df


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
    clicks_values = np.array(clicks_data.drop(labels = ['time_stamp'],  axis=1).sum(axis=1))
    
    sessions_values = np.array(sessions_data.drop(labels = ['time_stamp'],  axis=1).sum(axis=1))

    
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.plot(times,  clicks_values, label = 'Clicks')
    ax.plot(times,  sessions_values, label = "Sessions")
    
    ax.legend()
    ax.set_title(title)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{title}.png')

class PDF(FPDF):
    def __init__(self, timestamp):
        super().__init__()
        self.timestamp = timestamp
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        # Custom logo and positioning
        # Create an `assets` folder and put any wide and short image inside
        # Name the image `logo.png`
        # self.image('assets/scout-blue.png', 10, 8, 33)
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, f'Report - {datetime.datetime.fromtimestamp(self.timestamp).strftime("%B %d, %Y")}', 0, 0, 'R')
        self.ln(20)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_body(self, images):
        # Determine how many plots there are per page and set positions
        # and margins accordingly

        if len(images) == 3:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
            self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        elif len(images) == 2:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        else:
            self.image(images[0], 15, 25, self.WIDTH - 30)

    def print_page(self, images):
        # Generates the report
        self.add_page()
        self.page_body(images)

raw_stats_dir = 'rebrandly_raw_stats'

def make_pdf(raw_stats_dir, brand, campaign):
    
    data_list = os.listdir(os.path.join(raw_stats_dir, brand, campaign))
    timestamp = int(data_list[-1].split('.')[0])
    graphs_dir = f'{campaign}_graphs'

    clicks_df = aggregate_data(data_list, os.path.join(raw_stats_dir, brand, campaign), 'clicks')
    sessions_df = aggregate_data(data_list, os.path.join(raw_stats_dir, brand, campaign), 'sessions')

    sessions_and_clicks_over_time_title = f'Plot showing Clicks and Sessions over time for the {campaign} campaign'
    plot_sessions_and_clicks_over_time(clicks_df, sessions_df, sessions_and_clicks_over_time_title, graphs_dir)
    
    clicks_for_each_user_title = f'Plot showing Clicks over time for each influencer for the {campaign} campaign'
    plot_for_each_user(clicks_df, clicks_for_each_user_title, graphs_dir)
    
    sessions_for_each_user_title = f'Plot showing Sessions over time for each influencer for the {campaign} campaign'
    plot_for_each_user(sessions_df, sessions_for_each_user_title, graphs_dir)

    pdf = PDF(timestamp)

    pdf.print_page([f'./{graphs_dir}/{sessions_and_clicks_over_time_title}.png'])

    pdf.print_page([f'./{graphs_dir}/{clicks_for_each_user_title}.png'])

    pdf.print_page([f'./{graphs_dir}/{sessions_for_each_user_title}.png'])


    pdf.output(f'{brand}_{campaign}_{timestamp}.pdf', 'F')



make_pdf(raw_stats_dir, 'paycity', 'paycity_1')