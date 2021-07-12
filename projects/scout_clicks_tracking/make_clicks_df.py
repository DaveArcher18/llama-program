import os 

import numpy as np

import pandas as pd 

import datetime 

raw_data_dir = 'rebrandly_raw_stats'
#change raw_data_dir to change folder containing data

brands = os.listdir(raw_data_dir)

campaigns = []

brand_to_campaign_dict = {}
campaign_to_data_dict = {}


for i in brands:
    brand_to_campaign_dict[i] = os.listdir(os.path.join(raw_data_dir, i))
    for j in brand_to_campaign_dict[i]:
        campaign_to_data_dict[j] = os.listdir(os.path.join(raw_data_dir, i, j))

for i in brands:
    campaigns.append(brand_to_campaign_dict[i])

for i in brands:
   print(f'Brand: {i}')
   print(f'Campaign: {brand_to_campaign_dict[i]}')
   for j in brand_to_campaign_dict[i]:
       print(campaign_to_data_dict[j])



### working with paycity and campaign paycity_1
paycity_paycity1_data = campaign_to_data_dict[brand_to_campaign_dict['paycity'][0]]
corium_corium1_data = campaign_to_data_dict[brand_to_campaign_dict['corium'][0]]



def aggregate_data(data_list, filepath, column):
    '''Takes a list containing csvs, a filepath to the folder containing the csvs and a column name in the csv
    Returns a dataframe that has timestamp and slashtags as columns and the timestamps and column entires as rows.'''
    
    data0 = pd.read_csv(os.path.join(raw_data_dir, filepath, data_list[0]))
    timestamp0 = datetime.datetime.fromtimestamp(int(data_list[0].split('.')[0]))

    df = pd.DataFrame(columns = ['time_stamp'] + list(data0['slashtag']))

    df.loc[len(df.index)] = [timestamp0] + list(data0[column])

    for i in data_list[1:]:
        data = pd.read_csv(os.path.join(raw_data_dir, filepath, i))
        df.loc[len(df.index)] = [datetime.datetime.fromtimestamp(int(i.split('.')[0]))] + list(data[column])

    return df

paycity_1_clicks = aggregate_data(paycity_paycity1_data, 'paycity/paycity_1', 'clicks') 
paycity_1_clicks.to_csv('paycity_1_clicks.csv', index = False)

paycity_1_sessions = aggregate_data(paycity_paycity1_data, 'paycity/paycity_1', 'sessions')
paycity_1_sessions.to_csv('paycity_1_sessions.csv', index = False)

corium_1_clicks = aggregate_data(corium_corium1_data, 'corium/corium_1', 'clicks')
corium_1_clicks.to_csv('corium_1_clicks.csv', index = False)

corium_1_sessions = aggregate_data(corium_corium1_data, 'corium/corium_1', 'sessions')
corium_1_sessions.to_csv('corium_1_sessions.csv', index = False)



