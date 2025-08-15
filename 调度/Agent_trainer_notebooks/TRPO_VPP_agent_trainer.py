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
from sb3_contrib import MaskablePPO #The available algoritmhs in sb3-contrib for the custom environment with MultiInputPolicy
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

TRPO_path = "trained_models/TRPO_models/"

#In Colab, uncomment below:
#%env "WANDB_NOTEBOOK_NAME" "Agent_trainer_notebooks/TRPO_VPP_Hyperp_Sweep.ipynb"
os.environ['WANDB_NOTEBOOK_NAME'] = 'Agent_trainer_notebooks/TRPO_VPP_Hyperp_Sweep.ipynb'
wandb.login(relogin=True)

#In local notebook, uncomment below:
#your_wandb_login_code = 0123456789abcdefghijklmnopqrstwxyzàèìòù0 #example length
#!wandb login {your_wandb_login_code}

#wandb model configuration
config = {
    "policy_type": "MultiInputPolicy",
    "n_steps": 17520,
    "batch_size": 17520,
    "n_epochs": 15,
    #"total_timesteps": 10512300,
    "total_timesteps": 10512300,
    "learning_rate": 0.0009236773354410264,
    "gamma": 0.9194093400432,
    "gae_lambda": 0.92,
    "clip_range": 0.4,
    "ent_coef": 2.7839056618363775e-10,
    "vf_coef": 0.33358952685840515,
    "ortho_init": True,
    "activation_fn": torch.nn.modules.activation.Tanh,
    "optimizer_class": torch.optim.RMSprop,
    "net_arch": [dict(pi=[128, 128], vf=[128, 128])],  #[256, dict(pi=[512, 512], vf=[512, 512])]
    #"net_arch": [128, dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024])]
    "normalize_advantage": True,
            #"values":  [True, False]
    "max_grad_norm": 1,
            #"values": [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
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
# ENVIRONMENT WRAPPING
X_env = Monitor(env)
# Vectorized environment wrapper
X_env = DummyVecEnv([lambda: X_env])

# Sync custom tensorboard patch
# wandb.tensorboard.patch(root_logdir=wandb.run.dir, pytorch=True)

# model = A2C(config["policy_type"], X_env, verbose=1)
policy_kwargs = dict(
    ortho_init=config["ortho_init"],
    net_arch=config["net_arch"],
    activation_fn=config["activation_fn"],
    optimizer_class=config["optimizer_class"]
)

# model definition
model = TRPO(config["policy_type"], X_env,
             learning_rate=config["learning_rate"],
             n_steps=config["n_steps"],
             batch_size=config["batch_size"],
             n_epochs=config["n_epochs"],
             gamma=config["gamma"],
             gae_lambda=config["gae_lambda"],
             clip_range=config["clip_range"],
             ent_coef=config["ent_coef"],
             vf_coef=config["vf_coef"],

             normalize_advantage=config["normalize_advantage"],
             max_grad_norm=config["max_grad_norm"],

             # create_eval_env = False,
             policy_kwargs=policy_kwargs,
             verbose=0,
             tensorboard_log=f"wandb/tensorboard_log/"
             # tensorboard_log= wandb.run.dir
             )

# wandb.watch(model)



model.learn(total_timesteps=config["total_timesteps"],
    tb_log_name='TRPO',
    callback=WandbCallback(
        gradient_save_freq=10000,
        #model_save_path=f"trained_models/{run.id}",
        verbose=1,
    ))

#wandb.save(f"model.{run.id}")
model.save(current_folder + TRPO_path + f"model_TRPO_{run.id}")
model.save(os.path.join(wandb.run.dir, f"model_TRPO_{run.id}"))
wandb.save(f"model_TRPO_{run.id}")
#wandb.save(f'wandb/tensorboard_log/TRPO_{run.id}_1')

#EVALUATION of the trained model
cumulative_reward, std_reward = evaluate_policy(model, X_env, n_eval_episodes=1, render=False)
print(f"Average reward: {cumulative_reward}, St.dev: {std_reward}")

#TEST Model
episodes = 1
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        #env.render()
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks) #Now using our trained model with deterministic prediction [should improve performances]
        #action, _ = model.predict(obs) #Now using the trained model without deterministic prediction
        obs, reward, done, info = env.step(action)
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

env.plot_EVs_kpi()

env.plot_load_kpi()


#implement loading VPP_results_plot as artifact


#plot_VPP_input_data = env.plot_VPP_input_data()
#plot_VPP_input_data.show()

env.close()
run.finish()
wandb.finish()


