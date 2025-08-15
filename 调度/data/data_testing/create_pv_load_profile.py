#Moving to the current directory
current_folder = ''

input_folder = current_folder + 'raw_datasets/'
output_folder = current_folder + 'scenario_datasets/'

import numpy as np
import pandas as pd
import plotly.express as px

pv_data = pd.read_csv(input_folder + 'ninja_pv_52.5170_13.3889_corrected_2020_MERRA_Berlin.csv')
pv_data = pv_data.drop(['local_time'], axis=1)
pv_data['time'] = pd.to_datetime(pv_data['time'])

index = len(pv_data['time'])
timestamp = pd.to_datetime("2021-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
last_value = pv_data['electricity'][index-1]
new_row = pd.DataFrame([[timestamp, last_value]], columns=["time",'electricity'], index=[index])
pv_data = pd.concat([pv_data, pd.DataFrame(new_row)], ignore_index=False)
pv_data.rename(columns = {'electricity':'solar_power'}, inplace = True)

#Scaling capacity of PV
pv_data['solar_power'] = pv_data['solar_power']*16
pv_data.info()

pv_data_15 = pv_data.resample('15min', on='time').agg({'time':'min','solar_power':'min'})
pv_data_15['time'] = pv_data_15.index.values
pv_data_15['solar_power'].fillna(method='ffill', inplace=True)
pv_data_15 = pv_data_15.drop(['time'], axis=1)

pv_data_ = pv_data_15.loc['2020-01-01 00:00:00':'2021-01-01 00:00:00']
pv_data_.info()

pv_data_csv = pv_data_.to_csv(output_folder + 'PV_load_2020_profile.csv', index = True)
#pv_data_.iplot(title='PV Solar power', yTitle='kW')
px.line(pv_data_["solar_power"], title=('PV Solar power'), color_discrete_sequence=["orange"]).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2020-06-01 00:00:00", "2020-07-01 00:00:00"])