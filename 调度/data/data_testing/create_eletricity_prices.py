#Moving to the project directory
from google.colab import drive
drive.mount('/content/drive')
#Moving to the current directory
current_folder = ''

input_folder = current_folder + 'raw_datasets/'
output_folder = current_folder + 'scenario_datasets/'

import numpy as np
import pandas as pd
import plotly.express as px

prices_data = pd.read_csv(input_folder + 'Day-ahead Prices_202001010000-202101010000.csv')

#prices_data.head()
prices_data.info()

prices_data["MTU (CET/CEST)"] = prices_data["MTU (CET/CEST)"].str.slice(start=0, stop=16)
prices_data.rename(columns = {'MTU (CET/CEST)':'time'}, inplace = True)
prices_data['time'] = pd.to_datetime(prices_data['time'])

#COnversion from EUR/MWh to EUR/kWh
prices_data.rename(columns = {'Day-ahead Price [EUR/MWh]':'EUR/kWh'}, inplace = True)
prices_data['EUR/kWh'] = pd.to_numeric(prices_data['EUR/kWh'],errors='coerce') / 1000

prices_data = prices_data.drop(['BZN|DE-LU'], axis=1)
prices_data = prices_data.drop(['Currency'], axis=1)

index = len(prices_data['time'])
timestamp = pd.to_datetime("2021-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
last_value = prices_data['EUR/kWh'][index-1]
new_row = pd.DataFrame([[timestamp, last_value]], columns=["time",'EUR/kWh'], index=[index])
prices_data = pd.concat([prices_data, pd.DataFrame(new_row)], ignore_index=False)

prices_data.tail()

prices_data = prices_data.resample('15min', on='time').agg({'time':'min','EUR/kWh':'min'})

#prices_data['time'] = prices_data.index.values
prices_data['EUR/kWh'].fillna(method='ffill', inplace=True)
prices_data = prices_data.loc['2020-01-01 00:00:00':'2021-01-01 00:00:00']
prices_data = prices_data.drop(['time'], axis=1)
prices_data.info()

prices_csv = prices_data.to_csv(output_folder + 'market_prices_2020_profile.csv', index = True)
px.line(prices_data['EUR/kWh'], title=('Electricity prices')).update_layout(yaxis_title="€/kWh", xaxis_rangeslider_visible=True, xaxis_range=["2020-06-01 00:00:00", "2020-07-01 00:00:00"])