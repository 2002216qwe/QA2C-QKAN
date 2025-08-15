#Moving to the current directory
current_folder = ''

input_folder = current_folder + 'raw_datasets/'
output_folder = current_folder + 'scenario_datasets/'

import numpy as np
import pandas as pd
import plotly.express as px

wt_data = pd.read_csv(input_folder + 'ninja_wind_52.5170_13.3889_corrected_2018_MERRA_Berlin.csv')
wt_data = wt_data.drop(['local_time'], axis=1)
wt_data['time'] = pd.to_datetime(wt_data['time'])

index = len(wt_data['time'])
timestamp = pd.to_datetime("2019-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
last_value = wt_data['electricity'][index-1]
new_row = pd.DataFrame([[timestamp, last_value]], columns=["time",'electricity'], index=[index])
wt_data = pd.concat([wt_data, pd.DataFrame(new_row)], ignore_index=False)
wt_data.rename(columns = {'electricity':'wind_power'}, inplace = True)

#Scaling capacity of WT
wt_data['wind_power'] = wt_data['wind_power']*12
wt_data.info()

wt_data_15 = wt_data.resample('15min', on='time').agg({'time':'min','wind_power':'min'})
wt_data_15['time'] = wt_data_15.index.values
wt_data_15['wind_power'].fillna(method='ffill', inplace=True)
wt_data_15 = wt_data_15.drop(['time'], axis=1)

wt_data_ = wt_data_15.loc['2018-01-01 00:00:00':'2019-01-01 00:00:00']
wt_data_.tail()

wt_data_csv = wt_data_.to_csv(output_folder + 'WT_load_2018_profile.csv', index = True)
#wt_data_.iplot(title='WT wind power', yTitle='kW', colors='#3aa2cd')
px.line(wt_data_["wind_power"], title=('WT wind power'), color_discrete_sequence=["#3aa2cd"]).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2018-06-01 00:00:00", "2018-07-01 00:00:00"])

