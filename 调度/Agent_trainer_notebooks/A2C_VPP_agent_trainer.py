import yaml
import torch
from VPP_environment import VPPEnv, VPP_Scenario_config
from elvis.config import ScenarioConfig
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C #The available algoritmhs in sb3-contrib for the custom environment with MultiInputPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
import random

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
VPP_data_input_path = current_folder + 'data/data_training/environment_table/' + 'Environment_data_2019.csv'
elvis_input_folder = current_folder + 'data/config_builder/'

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

print(elvis_config_file)
print(VPP_config_file)

#Environment initialization
env = VPPEnv(VPP_data_input_path, elvis_config_file, VPP_config_file)
env.plot_ELVIS_data()

#Function to check custom environment and output additional warnings if needed
check_env(env)
env.plot_reward_functions()

A2C_path = "trained_models/A2C_models/"

#In Colab, uncomment below:
#%env "WANDB_DISABLE_CODE" True
#%env "WANDB_NOTEBOOK_NAME" "Agent_trainer_notebooks/A2C_VPP_agent_trainer.ipynb"
os.environ['WANDB_NOTEBOOK_NAME'] = 'Agent_trainer_notebooks/A2C_VPP_agent_trainer.ipynb'
wandb.login(relogin=True)

#In local notebook, uncomment below:
#your_wandb_login_code = 0123456789abcdefghijklmnopqrstwxyzàèìòù0 #example length
#!wandb login {your_wandb_login_code}


#wandb model configuration
config = {
    "policy_type": "MultiInputPolicy",
    "n_steps": 8760,
    "batch_size": 8760,
    "total_timesteps": 1000000,
    "learning_rate": 0.0007145030954379823,
    "gamma": 0.9159078953021682,
    "gae_lambda": 0.8,
    #"clip_range": 0.4,
    "ent_coef": 1.5005326968113368e-7,
    "vf_coef": 0.011059086790668691,
    "ortho_init": True,
    "activation_fn": torch.nn.modules.activation.Tanh,
    "optimizer_class": torch.optim.RMSprop,
    "net_arch": [64, dict(pi=[256, 256], vf=[256, 256])],
    "use_rms_prop": False,
    "normalize_advantage": True,
            #"values":  [True, False]
    "max_grad_norm": 0.7,
}

#wandb.tensorboard.patch(root_logdir="log_path")
run = wandb.init(
    project="RL_VPP_Thesis",
    #entity="user_avocado",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=False # optional
)

#ENVIRONMENT WRAPPING
X_env = Monitor(env)
#Vectorized environment wrapper
X_env = DummyVecEnv([lambda: X_env])

#Sync custom tensorboard patch
#wandb.tensorboard.patch(root_logdir=wandb.run.dir, pytorch=True)
tensorboard_log_path = "wandb/tensorboard_log/"

#model = A2C(config["policy_type"], X_env, verbose=1)
policy_kwargs =  dict(
            ortho_init = config["ortho_init"],
            net_arch = config["net_arch"],
            activation_fn = config["activation_fn"],
            #optimizer_class = config["optimizer_class"]
        )

#model definition
model = A2C(config["policy_type"], X_env,
                    learning_rate = config["learning_rate"],
                    n_steps = config["n_steps"],
                    #batch_size = batch_size,
                    #n_epochs = config["n_epochs"],
                    gamma = config["gamma"],
                    gae_lambda = config["gae_lambda"],
                    #clip_range = config["clip_range"],
                    ent_coef = config["ent_coef"],
                    vf_coef = config["vf_coef"],
                    normalize_advantage = config["normalize_advantage"],
                    max_grad_norm = config["max_grad_norm"],
                    use_rms_prop = config["use_rms_prop"],
                    #create_eval_env = False,
                    policy_kwargs = policy_kwargs,
                    verbose=0,
                    #tensorboard_log= os.path.join(tensorboard_log_path,f'A2C_{run.id}_1')
                    tensorboard_log = tensorboard_log_path
                    )

#wandb.watch(model)

#%%time

model.learn(total_timesteps=config["total_timesteps"],
    tb_log_name='A2C',
    callback=WandbCallback(
        gradient_save_freq=10000,
        #model_save_path=f"trained_models/{run.id}",
        verbose=1)
    )

#wandb.save(f"model.{run.id}")
model.save(current_folder + A2C_path + f"model_A2C_{run.id}")
model.save(os.path.join(wandb.run.dir, f"model_A2C_{run.id}"))
wandb.save(f"model_A2C_{run.id}")
#wandb.save(f'wandb/tensorboard_log/A2C_{run.id}_1')

#EVALUATION of the trained model
cumulative_reward, std_reward = evaluate_policy(model, X_env, n_eval_episodes=1, render=False)
print(f"Average reward: {cumulative_reward}, St.dev: {std_reward}")

#TEST Model
episodes = 1
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = [True]
    while not done:
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic = True) #Now using our trained model with deterministic prediction [should improve performances]
        #env.lstm_state = lstm_states
        obs, reward, done, info = env.step(action)
        episode_starts = done
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))

VPP_table = env.VPP_table

env.plot_VPP_energies()

env.plot_Elvis_results()

env.plot_VPP_results()

env.plot_VPP_supply_demand()

env.plot_rewards_stats()

env.plot_rewards_results()

env.plot_VPP_Elvis_comparison()

env.plot_actions_kpi()

env.plot_EVs_kpi()

env.plot_load_kpi()

env.plot_EVs_kpi()

env.plot_yearly_load_log()

#implement loading VPP_results_plot as artifact

env.close()
run.finish()
wandb.finish()