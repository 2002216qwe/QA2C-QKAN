import pandas as pd
import numpy as np
import plotly.express as px


#Cloning repository and changing directory
#!git clone https://github.com/francescomaldonato/RL_VPP_Thesis.git
#%cd RL_VPP_Thesis/data/
#%ls

#Moving to the project directory
current_folder = ''

input_folder = current_folder + 'data_training/scenario_datasets/'
output_folder = current_folder + 'data_training/environment_table/'

market_prices = pd.read_csv(input_folder + 'market_prices_2019_profile.csv')
PV_load = pd.read_csv(input_folder + 'PV_load_2019_profile.csv')
WT_load = pd.read_csv(input_folder + 'WT_load_2019_profile.csv')
household_load = pd.read_csv(input_folder + 'households_load_profile.csv')

#Time indexes check
time_serie = market_prices["time"].values

assert(all(time_serie == PV_load["time"].values))
assert(all(time_serie == WT_load["time"].values))

#initialization of VPP table
VPP_table = pd.DataFrame({"time": time_serie})
VPP_table["time"] = pd.to_datetime(VPP_table["time"])

print(f"Total timesteps: {len(time_serie)}")

#Data preparation
household_load.rename(columns = {'power':'household_power'}, inplace = True) #kW
household_load.tail()

household_load["time"] = market_prices["time"]
assert(all(time_serie == household_load["time"].values))
household_load.info()

#Merging of tables
VPP_table = pd.concat((VPP_table, household_load["household_power"], PV_load["solar_power"], WT_load["wind_power"], market_prices["EUR/kWh"] ), axis = 1)
VPP_table["renewable_power"] = VPP_table["solar_power"] + VPP_table["wind_power"]
VPP_table["House&RW_load"] = VPP_table["household_power"] - VPP_table["renewable_power"] #kW
VPP_table["total_cost"] = VPP_table["House&RW_load"] * VPP_table["EUR/kWh"] / 4

VPP_table = VPP_table.set_index("time")
VPP_table.info()
VPP_table.head(5)

print("Energy consumed: kW", VPP_table["household_power"].sum()/4)
#VPP_table["household_power"].iplot(title='Households power', yTitle='kW', color='red')
px.area(VPP_table["household_power"], title=('Households power'), color_discrete_sequence=["red"]).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2019-06-01 00:00:00", "2019-07-01 00:00:00"])