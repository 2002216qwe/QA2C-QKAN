#Moving to current directory
current_folder = ''

input_folder = current_folder + 'raw_datasets/'
output_folder = current_folder + 'scenario_datasets/'

import numpy as np
import pandas as pd
import plotly.express as px

household_data = pd.read_csv(input_folder + 'consumer-00000015_glimpse_TEST_year.csv')

household_data['time'] = pd.to_datetime(household_data['time'])
household_data.info()

household_data = household_data.drop(['energy'], axis=1)
household_data.tail()

#Resampling time

time_sampled = household_data.resample('15min', on='time').time.sum
household_data = household_data.resample('15min', on='time').agg({'time':'min', 'power':'mean'})
household_data = household_data.drop(['time'], axis=1)
household_data = household_data.loc['2017-01-01 00:00:00':'2018-01-01 00:00:00']
household_data[household_data["power"]==""] = np.NaN
household_data.power = household_data.power.ffill()
#Convertion from mW to kW
household_data['power'] = household_data['power']/1000000
household_data.head()

household_data.info()

#Save and plot data
#household_csv = household_data.to_csv(output_folder + 'household_load_profile.csv', index = True)
px.line(household_data["power"], title=('Single household power consuption')).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2017-06-01 00:00:00", "2017-07-01 00:00:00"])