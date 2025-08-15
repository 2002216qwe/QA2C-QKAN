import yaml
import numpy as np
from gym import Env
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
case = 'wohnblock_household_simulation_adaptive_10.yaml' #(10 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_15.yaml' #(15 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_25.yaml' #(25 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_30.yaml' #(30 EVs arrivals per week with 50% average battery)
#case = 'wohnblock_household_simulation_adaptive_35.yaml' #(35 EVs arrivals per week with 50% average battery)

with open(elvis_input_folder + case, 'r') as file:
    yaml_str = yaml.full_load(file)

elvis_config_file = ScenarioConfig.from_yaml(yaml_str)
VPP_config_file = VPP_Scenario_config(yaml_str)

print(elvis_config_file)
print(VPP_config_file)

#TESTING Environment initialization
env = VPPEnv(VPP_testing_data_input_path, elvis_config_file, VPP_config_file)
#env.plot_VPP_input_data()

env.plot_ELVIS_data()