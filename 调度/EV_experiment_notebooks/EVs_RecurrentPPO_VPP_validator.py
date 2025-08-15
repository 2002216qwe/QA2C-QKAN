import yaml
import numpy as np
import pandas as pd
from VPP_environment import VPPEnv, VPP_Scenario_config
from elvis.config import ScenarioConfig
import os
import torch
import random
import wandb
from sb3_contrib import RecurrentPPO #The available algoritmhs in sb3-contrib for the custom environment with MultiInputPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env

#Check if cuda device is available for training
print("Torch-Cuda available device:", torch.cuda.is_available())
print(sb3.get_system_info())

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

#Loading paths for input data
current_folder = ''
VPP_training_data_input_path = current_folder + 'data/data_training/environment_table/' + 'Environment_data_2019.csv'
VPP_testing_data_input_path = current_folder + 'data/data_testing/environment_table/' + 'Environment_data_2020.csv'
VPP_validating_data_input_path = current_folder + 'data/data_validating/environment_table/' + 'Environment_data_2018.csv'
elvis_input_folder = current_folder + 'data/config_builder/'

#case = 'wohnblock_household_simulation_adaptive.yaml' #(loaded by default, 20 EVs arrivals per week with 50% average battery)

#Try different simulation parameters, uncomment below
#case = 'wohnblock_household_simulation_adaptive_10.yaml' #(10 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_15.yaml' #(15 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_25.yaml' #(25 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_30.yaml' #(30 EVs arrivals per week with 50% average battery)
case = 'wohnblock_household_simulation_adaptive_35.yaml' #(35 EVs arrivals per week with 50% average battery)

with open(elvis_input_folder + case, 'r') as file:
    yaml_str = yaml.full_load(file)

elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
VPP_config_file = VPP_Scenario_config(yaml_str)

print(elvis_config_file)
print(VPP_config_file)

#TESTING Environment initialization
env = VPPEnv(VPP_validating_data_input_path, elvis_config_file, VPP_config_file)
#env.plot_VPP_input_data()
#Function to check custom environment and output additional warnings if needed
check_env(env)
#env.plot_reward_functions()

#Loading training model, from local directory or from wandb previous trainings
RecurrentPPO_path = "trained_models/RecurrentPPO_models/model_RecurrentPPO_"

#model_id = "s37o8q0n"
model_id = "333ckz0i"
model = RecurrentPPO.load(RecurrentPPO_path + model_id, env=env)

# run_id_restore = "2y2dqvyn"
# model = wandb.restore(f'model_{run_id_restore}.zip', run_path=f"francesco_maldonato/RL_VPP_Thesis/{run_id_restore}")
results_table = []

#TEST Model
episodes = 10
results = []
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        #env.render()
        action_masks = get_action_masks(env)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True) #Now using our trained model with deterministic prediction [should improve performances]
        env.lstm_state = lstm_states
        obs, reward, done, info = env.step(action)
        episode_starts = done
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    results.append(env.quick_results)

stacked_results = np.stack(results)

results.append(np.array([str(env.EVs_n)+"_EVs_mean", np.mean(stacked_results[:, 1].astype('float32')), np.mean(stacked_results[:, 2].astype('float32')), np.mean(stacked_results[:, 3].astype('float32')), np.mean(stacked_results[:, 4].astype('float32')), np.mean(stacked_results[:, 5].astype('float32'))]))
results_table.extend(results)
#Save the VPP table
#VPP_table = env.save_VPP_table(save_path='data/environment_optimized_output/VPP_table.csv')
#VPP_table = env.VPP_table

#case = 'wohnblock_household_simulation_adaptive.yaml' #(loaded by default, 20 EVs arrivals per week with 50% average battery)

#Try different simulation parameters, uncomment below
#case = 'wohnblock_household_simulation_adaptive_10.yaml' #(10 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_15.yaml' #(15 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_25.yaml' #(25 EVs arrivals per week with 50% average battery)
case = 'wohnblock_household_simulation_adaptive_30.yaml' #(30 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_35.yaml' #(35 EVs arrivals per week with 50% average battery)

with open(elvis_input_folder + case, 'r') as file:
    yaml_str = yaml.full_load(file)

elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
VPP_config_file = VPP_Scenario_config(yaml_str)

#print(elvis_config_file)
#print(VPP_config_file)

#TESTING Environment initialization
env = VPPEnv(VPP_validating_data_input_path, elvis_config_file, VPP_config_file)
#Function to check custom environment and output additional warnings if needed
check_env(env)

model = RecurrentPPO.load(RecurrentPPO_path + model_id, env=env)

#TEST Model
episodes = 10
results = []
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        #env.render()
        action_masks = get_action_masks(env)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True) #Now using our trained model with deterministic prediction [should improve performances]
        env.lstm_state = lstm_states
        obs, reward, done, info = env.step(action)
        episode_starts = done
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    results.append(env.quick_results)

stacked_results = np.stack(results)
results.append(np.array([str(env.EVs_n)+"_EVs_mean", np.mean(stacked_results[:, 1].astype('float32')), np.mean(stacked_results[:, 2].astype('float32')), np.mean(stacked_results[:, 3].astype('float32')), np.mean(stacked_results[:, 4].astype('float32')), np.mean(stacked_results[:, 5].astype('float32'))]))
results_table.extend(results)
#Save the VPP table
#VPP_table = env.save_VPP_table(save_path='data/environment_optimized_output/VPP_table.csv')
#VPP_table = env.VPP_table

#case = 'wohnblock_household_simulation_adaptive.yaml' #(loaded by default, 20 EVs arrivals per week with 50% average battery)

#Try different simulation parameters, uncomment below
#case = 'wohnblock_household_simulation_adaptive_10.yaml' #(10 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_15.yaml' #(15 EVs arrivals per week with 50% average battery)
case = 'wohnblock_household_simulation_adaptive_25.yaml' #(25 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_30.yaml' #(30 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_35.yaml' #(35 EVs arrivals per week with 50% average battery)

with open(elvis_input_folder + case, 'r') as file:
    yaml_str = yaml.full_load(file)

elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
VPP_config_file = VPP_Scenario_config(yaml_str)

#print(elvis_config_file)
#print(VPP_config_file)

#TESTING Environment initialization
env = VPPEnv(VPP_validating_data_input_path, elvis_config_file, VPP_config_file)
#Function to check custom environment and output additional warnings if needed
check_env(env)

model = RecurrentPPO.load(RecurrentPPO_path + model_id, env=env)

#TEST Model
episodes = 10
results = []
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        #env.render()
        action_masks = get_action_masks(env)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True) #Now using our trained model with deterministic prediction [should improve performances]
        env.lstm_state = lstm_states
        obs, reward, done, info = env.step(action)
        episode_starts = done
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    results.append(env.quick_results)

stacked_results = np.stack(results)
results.append(np.array([str(env.EVs_n)+"_EVs_mean", np.mean(stacked_results[:, 1].astype('float32')), np.mean(stacked_results[:, 2].astype('float32')), np.mean(stacked_results[:, 3].astype('float32')), np.mean(stacked_results[:, 4].astype('float32')), np.mean(stacked_results[:, 5].astype('float32'))]))
results_table.extend(results)
#Save the VPP table
#VPP_table = env.save_VPP_table(save_path='data/environment_optimized_output/VPP_table.csv')
#VPP_table = env.VPP_table

case = 'wohnblock_household_simulation_adaptive.yaml' #(loaded by default, 20 EVs arrivals per week with 50% average battery)

#Try different simulation parameters, uncomment below
#case = 'wohnblock_household_simulation_adaptive_10.yaml' #(10 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_15.yaml' #(15 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_25.yaml' #(25 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_30.yaml' #(30 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_35.yaml' #(35 EVs arrivals per week with 50% average battery)

with open(elvis_input_folder + case, 'r') as file:
    yaml_str = yaml.full_load(file)

elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
VPP_config_file = VPP_Scenario_config(yaml_str)

#print(elvis_config_file)
#print(VPP_config_file)

#TESTING Environment initialization
env = VPPEnv(VPP_validating_data_input_path, elvis_config_file, VPP_config_file)
#Function to check custom environment and output additional warnings if needed
check_env(env)

model = RecurrentPPO.load(RecurrentPPO_path + model_id, env=env)

#TEST Model
episodes = 10
results = []
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        #env.render()
        action_masks = get_action_masks(env)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True) #Now using our trained model with deterministic prediction [should improve performances]
        env.lstm_state = lstm_states
        obs, reward, done, info = env.step(action)
        episode_starts = done
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    results.append(env.quick_results)

stacked_results = np.stack(results)
results.append(np.array([str(env.EVs_n)+"_EVs_mean", np.mean(stacked_results[:, 1].astype('float32')), np.mean(stacked_results[:, 2].astype('float32')), np.mean(stacked_results[:, 3].astype('float32')), np.mean(stacked_results[:, 4].astype('float32')), np.mean(stacked_results[:, 5].astype('float32'))]))
results_table.extend(results)
#Save the VPP table
#VPP_table = env.save_VPP_table(save_path='data/environment_optimized_output/VPP_table.csv')
#VPP_table = env.VPP_table

#case = 'wohnblock_household_simulation_adaptive.yaml' #(loaded by default, 20 EVs arrivals per week with 50% average battery)

#Try different simulation parameters, uncomment below
#case = 'wohnblock_household_simulation_adaptive_10.yaml' #(10 EVs arrivals per week with 50% average battery)
case = 'wohnblock_household_simulation_adaptive_15.yaml' #(15 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_25.yaml' #(25 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_30.yaml' #(30 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_35.yaml' #(35 EVs arrivals per week with 50% average battery)

with open(elvis_input_folder + case, 'r') as file:
    yaml_str = yaml.full_load(file)

elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
VPP_config_file = VPP_Scenario_config(yaml_str)

#print(elvis_config_file)
#print(VPP_config_file)

#TESTING Environment initialization
env = VPPEnv(VPP_validating_data_input_path, elvis_config_file, VPP_config_file)
#Function to check custom environment and output additional warnings if needed
check_env(env)

model = RecurrentPPO.load(RecurrentPPO_path + model_id, env=env)

#TEST Model
episodes = 10
results = []
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        #env.render()
        action_masks = get_action_masks(env)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True) #Now using our trained model with deterministic prediction [should improve performances]
        env.lstm_state = lstm_states
        obs, reward, done, info = env.step(action)
        episode_starts = done
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    results.append(env.quick_results)

stacked_results = np.stack(results)
results.append(np.array([str(env.EVs_n)+"_EVs_mean", np.mean(stacked_results[:, 1].astype('float32')), np.mean(stacked_results[:, 2].astype('float32')), np.mean(stacked_results[:, 3].astype('float32')), np.mean(stacked_results[:, 4].astype('float32')), np.mean(stacked_results[:, 5].astype('float32'))]))
results_table.extend(results)
#Save the VPP table
#VPP_table = env.save_VPP_table(save_path='data/environment_optimized_output/VPP_table.csv')
#VPP_table = env.VPP_table

#case = 'wohnblock_household_simulation_adaptive.yaml' #(loaded by default, 20 EVs arrivals per week with 50% average battery)

#Try different simulation parameters, uncomment below
case = 'wohnblock_household_simulation_adaptive_10.yaml' #(10 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_15.yaml' #(15 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_25.yaml' #(25 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_30.yaml' #(30 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_35.yaml' #(35 EVs arrivals per week with 50% average battery)

with open(elvis_input_folder + case, 'r') as file:
    yaml_str = yaml.full_load(file)

elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
VPP_config_file = VPP_Scenario_config(yaml_str)

#print(elvis_config_file)
#print(VPP_config_file)

#TESTING Environment initialization
env = VPPEnv(VPP_validating_data_input_path, elvis_config_file, VPP_config_file)
#Function to check custom environment and output additional warnings if needed
check_env(env)

model = RecurrentPPO.load(RecurrentPPO_path + model_id, env=env)

#TEST Model
episodes = 10
results = []
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while not done:
        #env.render()
        action_masks = get_action_masks(env)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True) #Now using our trained model with deterministic prediction [should improve performances]
        env.lstm_state = lstm_states
        obs, reward, done, info = env.step(action)
        episode_starts = done
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    results.append(env.quick_results)

stacked_results = np.stack(results)
results.append(np.array([str(env.EVs_n)+"_EVs_mean", np.mean(stacked_results[:, 1].astype('float32')), np.mean(stacked_results[:, 2].astype('float32')), np.mean(stacked_results[:, 3].astype('float32')), np.mean(stacked_results[:, 4].astype('float32')), np.mean(stacked_results[:, 5].astype('float32'))]))
results_table.extend(results)
#Save the VPP table
#VPP_table = env.save_VPP_table(save_path='data/environment_optimized_output/VPP_table.csv')
#VPP_table = env.VPP_table

dataframe_results = pd.DataFrame(np.stack(results_table))
dataframe_results = dataframe_results.rename(columns={0:"Name",1:"underconsume",2:"overconsume",3:"overcost",4:"av_EV_energy_left",5:"cumulative_reward"})

dataframe_results.to_csv("data/algorithms_results/algorithms_results_table/EV_experiments_validating.csv")

dataframe_results.head()
