import pandas as pd
import numpy as np
import plotly.express as px




#Cloning repository and changing directory
#!git clone https://github.com/francescomaldonato/RL_VPP_Thesis.git
#cd RL_VPP_Thesis/data/
#%ls

#Moving to the project directory
current_folder = ''

input_folder = current_folder + 'data_testing/scenario_datasets/'
output_folder = current_folder + 'data_testing/environment_table/'

market_prices = pd.read_csv(input_folder + 'market_prices_2020_profile.csv')
PV_load = pd.read_csv(input_folder + 'PV_load_2020_profile.csv')
WT_load = pd.read_csv(input_folder + 'WT_load_2020_profile.csv')
household_load = pd.read_csv(input_folder + 'households_load_profile.csv')

market_prices["time"] = pd.to_datetime(market_prices["time"])
market_prices = market_prices.set_index("time")
market_prices.drop(market_prices.loc["2020-02-29 00:00:00":"2020-02-29 23:45:00"].index, inplace=True)
#market_prices.info()
PV_load["time"] = pd.to_datetime(PV_load["time"])
PV_load = PV_load.set_index("time")
PV_load.drop(PV_load.loc["2020-02-29 00:00:00":"2020-02-29 23:45:00"].index, inplace=True)
#PV_load.info()
WT_load["time"] = pd.to_datetime(WT_load["time"])
WT_load = WT_load.set_index("time")
WT_load.drop(WT_load.loc["2020-02-29 00:00:00":"2020-02-29 23:45:00"].index, inplace=True)
WT_load.info()


#Time indexes check
time_serie = market_prices.index

assert(all(time_serie == PV_load.index))
assert(all(time_serie == WT_load.index))

#initialization of VPP table
# VPP_table = pd.DataFrame({"time": time_serie})
# VPP_table["time"] = pd.to_datetime(VPP_table["time"])

print(f"Total timesteps: {len(time_serie)}")
# VPP_table.info()

#Data preparation
household_load.rename(columns = {'power':'household_power'}, inplace = True) #kW
household_load["time"] = market_prices.index
assert(all(time_serie == household_load["time"].values))
household_load = household_load.set_index("time")
household_load.info()

#Merging of tables
VPP_table = pd.concat((household_load["household_power"], PV_load["solar_power"], WT_load["wind_power"], market_prices["EUR/kWh"] ), axis = 1)
VPP_table["renewable_power"] = VPP_table["solar_power"] + VPP_table["wind_power"]
VPP_table["House&RW_load"] = VPP_table["household_power"] - VPP_table["renewable_power"] #kW
VPP_table["total_cost"] = VPP_table["House&RW_load"] * VPP_table["EUR/kWh"] / 4

#VPP_table = VPP_table.set_index("time")
VPP_table.info()
VPP_table.head(5)

print("Energy consumed: kW", VPP_table["household_power"].sum()/4)
#VPP_table["household_power"].iplot(title='Households power', yTitle='kW', color='red')
px.area(VPP_table["household_power"], title=('Households power'), color_discrete_sequence=["red"]).update_layout(yaxis_title="kW", xaxis_rangeslider_visible=True, xaxis_range=["2020-06-01 00:00:00", "2020-07-01 00:00:00"])
