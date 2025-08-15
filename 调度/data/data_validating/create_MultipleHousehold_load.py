#Moving to the current directory
current_folder = ''

input_folder = current_folder + 'raw_datasets/households_inputs_fromR_Blem/'
output_folder = current_folder + 'scenario_datasets/'

import numpy as np
import pandas as pd
import plotly.express as px

household_data1 = pd.read_csv(input_folder + 'consumer-00000011_glimpse_TEST_year.csv')
household_data2 = pd.read_csv(input_folder + 'consumer-00000012_glimpse_TEST_year.csv')
household_data3 = pd.read_csv(input_folder + 'consumer-00000013_glimpse_TEST_year.csv')
household_data4 = pd.read_csv(input_folder + 'consumer-00000014_glimpse_TEST_year.csv')

household_data1['time'] = pd.to_datetime(household_data1['time'])
household_data2['time'] = pd.to_datetime(household_data2['time'])
household_data3['time'] = pd.to_datetime(household_data3['time'])
household_data4['time'] = pd.to_datetime(household_data4['time'])

household_data4.info()

household_data1 = household_data1.drop(['energy'], axis=1)
household_data2 = household_data2.drop(['energy'], axis=1)
household_data3 = household_data3.drop(['energy'], axis=1)
household_data4 = household_data4.drop(['energy'], axis=1)
household_data4.tail()

#Resampling time

time_sampled1 = household_data1.resample('15min', on='time').time.sum
time_sampled2 = household_data2.resample('15min', on='time').time.sum
time_sampled3 = household_data3.resample('15min', on='time').time.sum
time_sampled4 = household_data4.resample('15min', on='time').time.sum

household_data1 = household_data1.resample('15min', on='time').agg({'time':'min', 'power':'mean'})
household_data2 = household_data2.resample('15min', on='time').agg({'time':'min', 'power':'mean'})
household_data3 = household_data3.resample('15min', on='time').agg({'time':'min', 'power':'mean'})
household_data4 = household_data4.resample('15min', on='time').agg({'time':'min', 'power':'mean'})

household_data1 = household_data1.drop(['time'], axis=1)
household_data2 = household_data2.drop(['time'], axis=1)
household_data3 = household_data3.drop(['time'], axis=1)
household_data4 = household_data4.drop(['time'], axis=1)

household_data1 = household_data1.loc['2017-01-01 00:00:00':'2018-01-01 00:00:00']
household_data2 = household_data2.loc['2017-01-01 00:00:00':'2018-01-01 00:00:00']
household_data3 = household_data3.loc['2017-01-01 00:00:00':'2018-01-01 00:00:00']
household_data4 = household_data4.loc['2017-01-01 00:00:00':'2018-01-01 00:00:00']

household_data1[household_data1["power"]==""] = np.NaN
household_data2[household_data2["power"]==""] = np.NaN
household_data3[household_data3["power"]==""] = np.NaN
household_data4[household_data4["power"]==""] = np.NaN

household_data1.power = household_data1.power.ffill()
household_data2.power = household_data2.power.ffill()
household_data3.power = household_data3.power.ffill()
household_data4.power = household_data4.power.ffill()

#Convertion from mW to kW
household_data1['power'] = household_data1['power']/1000000
household_data2['power'] = household_data2['power']/1000000
household_data3['power'] = household_data3['power']/1000000
household_data4['power'] = household_data4['power']/1000000

household_data4.info()

mean_housepower_4 = np.mean(household_data4["power"].values)
max_housepower_4 = household_data4["power"].values[np.argmax(household_data4["power"].values)]
min_housepower_4 = household_data4["power"].values[np.argmin(household_data4["power"].values)]
sum_housepower_4 = np.sum((household_data4["power"].values)/4)

print(f"Mean power rate: {round(mean_housepower_4,2)} kW, " +
        f"Max_power_rate: {round(max_housepower_4,2)} kW, " +
        f"Min_power_rate: {round(min_housepower_4,2)} kW, " +
        f"Total_energy_consumed: {round(sum_housepower_4,2)} kWh")
#household_data4.iplot(title="Household_4 power consuption", yTitle='kW')
px.line(household_data4["power"], title=('Household_4 power consuption')).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2017-06-01 00:00:00", "2017-07-01 00:00:00"])

mean_housepower_3 = np.mean(household_data3["power"].values)
max_housepower_3 = household_data3["power"].values[np.argmax(household_data3["power"].values)]
min_housepower_3 = household_data3["power"].values[np.argmin(household_data3["power"].values)]
sum_housepower_3 = np.sum((household_data3["power"].values)/4)

print(f"Mean power rate: {round(mean_housepower_3,2)} kW, " +
        f"Max_power_rate: {round(max_housepower_3,2)} kW, " +
        f"Min_power_rate: {round(min_housepower_3,2)} kW, " +
        f"Total_energy_consumed: {round(sum_housepower_3,2)} kWh")
#household_data3.iplot(title="Household_3 power consuption", yTitle='kW')
px.line(household_data3["power"], title=('Household_3 power consuption')).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2017-06-01 00:00:00", "2017-07-01 00:00:00"])

mean_housepower_2 = np.mean(household_data2["power"].values)
max_housepower_2 = household_data2["power"].values[np.argmax(household_data2["power"].values)]
min_housepower_2 = household_data2["power"].values[np.argmin(household_data2["power"].values)]
sum_housepower_2 = np.sum((household_data2["power"].values)/4)

print(f"Mean power rate: {round(mean_housepower_2,2)} kW, " +
        f"Max_power_rate: {round(max_housepower_2,2)} kW, " +
        f"Min_power_rate: {round(min_housepower_2,2)} kW, " +
        f"Total_energy_consumed: {round(sum_housepower_2,2)} kWh")
#household_data2.iplot(title="Household_2 power consuption", yTitle='kW')
px.line(household_data2["power"], title=('Household_2 power consuption')).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2017-06-01 00:00:00", "2017-07-01 00:00:00"])

mean_housepower_1 = np.mean(household_data1["power"].values)
max_housepower_1 = household_data1["power"].values[np.argmax(household_data1["power"].values)]
min_housepower_1 = household_data1["power"].values[np.argmin(household_data1["power"].values)]
sum_housepower_1 = np.sum((household_data1["power"].values)/4)

print(f"Mean power rate: {mean_housepower_1} kW, Max_power_rate: {max_housepower_1} kW, Total_energy_consumed: {sum_housepower_1} kWh")
print(f"Mean power rate: {round(mean_housepower_1,2)} kW, " +
        f"Max_power_rate: {round(max_housepower_1,2)} kW, " +
        f"Min_power_rate: {round(min_housepower_1,2)} kW, " +
        f"Total_energy_consumed: {round(sum_housepower_1,2)} kWh")
#household_data1.iplot(title="Household_1 power consuption", yTitle='kW')
px.line(household_data1["power"], title=('Household_1 power consuption')).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2017-06-01 00:00:00", "2017-07-01 00:00:00"])

household_data = household_data1
household_data['power'] = household_data1['power'] + household_data2['power'] + household_data3['power'] + household_data4['power']
household_data.info()

household_csv = household_data.to_csv(output_folder + 'households_load_profile.csv', index = True)

av_max_powers = np.mean([max_housepower_1, max_housepower_2, max_housepower_3, max_housepower_4])
mean_multi_housepower = np.mean(household_data["power"].values)

max_multi_housepower = household_data["power"].values[np.argmax(household_data["power"].values)]
min_multi_housepower = household_data["power"].values[np.argmin(household_data["power"].values)]
mean_active_housepower = (max_multi_housepower + min_multi_housepower)/2

sum_multi_housepower = np.sum((household_data["power"].values)/4)
print(f"Mean active_power: {round(mean_active_housepower,2)} kW")
print(f"Mean power rate: {round(mean_multi_housepower,2)} kW, " +
        f"Max_power_rate: {round(max_multi_housepower,2)} kW, " +
        f"Min_power_rate: {round(min_multi_housepower,2)} kW, " +
        f"Av.HouseMax_power_rate: {round(av_max_powers,2)} kW, " +
        f"Total_energy_consumed: {round(sum_multi_housepower,2)} kWh")
#household_data.iplot(title="Multiple household power consuption(4)", yTitle='kW')
px.line(household_data["power"], title=('Multiple household power consuption(4)')).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2017-06-01 00:00:00", "2017-07-01 00:00:00"])

