import os
import random
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from training.double_dqn import DoubleDQN
from stable_baselines3.common.monitor import Monitor
from envs.ids_env import IDSEnv
from training.hyperparams import (
    PPO_PARAMS, DQN_PARAMS, DDQN_PARAMS,
    TOTAL_TIMESTEPS, SEEDS, DATASETS, AGENTS,
    TIMESTEPS_PER_DATASET
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RewardLoggerCallback(BaseCallback):
    """
    Callback que registra la recompensa acumulada por episodio durante el entrenamiento.
    Al finalizar guarda un CSV con las curvas de convergencia.
    """

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.timestep_at_episode_end = []
        self.reward_current_episode = 0.0

    def _on_step(self):

        infos = self.locals.get('infos', [{}])
        if 'episode' in infos[0]:
            ep_reward = infos[0]['episode']['r']
            self.episode_rewards.append(ep_reward)
            self.timestep_at_episode_end.append(self.num_timesteps)

        return True

    def _on_training_end(self):

        convergence_data = pd.DataFrame({
            'episode':  range(1, len(self.episode_rewards) + 1),
            'reward':   self.episode_rewards,
            'timestep': self.timestep_at_episode_end
        })
        convergence_data.to_csv(self.log_path, index=False)
        print(f"  curve -> {self.log_path}")

def set_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(agent_name, env, seed, params):
    model_args = dict(
        policy='MlpPolicy',
        env=env,
        seed=seed,
        verbose=0,

        device='cpu',
        **params
    )

    if agent_name == 'PPO':
        return PPO(**model_args)
    elif agent_name == 'DDQN':

        return DoubleDQN(**model_args)
    else:
        return DQN(**model_args)

def get_params(agent_name):
    params_map = {
        'PPO':  PPO_PARAMS,
        'DQN':  DQN_PARAMS,
        'DDQN': DDQN_PARAMS
    }
    return params_map[agent_name]

def train_single_run(agent_name, dataset_name, seed, X_train, y_train, timesteps=None):
    run_id = f"{agent_name}_{dataset_name.upper()}_seed{seed}"

    model_path = os.path.join(BASE_DIR, 'models', run_id)
    curve_path = os.path.join(BASE_DIR, 'results', 'convergence_curves', f'{run_id}_curve.csv')

    print(f"\n{'='*55}")
    print(f"Entrenando: {run_id}")
    print(f"{'='*55}")

    set_seeds(seed)

    env = Monitor(IDSEnv(X_train.values, y_train.values))

    model = get_model(agent_name, env, seed, get_params(agent_name))

    reward_logger = RewardLoggerCallback(log_path=curve_path)

    if timesteps is not None:
        total_timesteps = timesteps
    else:
        total_timesteps = TIMESTEPS_PER_DATASET[dataset_name]

    model.learn(
        total_timesteps=total_timesteps,
        callback=reward_logger,
        progress_bar=True,
        reset_num_timesteps=True
    )

    model.save(model_path)
    print(f"  model -> {model_path}.zip")

    env.close()
    return model_path, curve_path

def run_all_experiments(datasets_dict):
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results', 'convergence_curves'), exist_ok=True)

    run_log = []

    for agent in AGENTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                X_train, X_test, y_train, y_test = datasets_dict[dataset]

                model_path, curve_path = train_single_run(
                    agent_name=agent,
                    dataset_name=dataset,
                    seed=seed,
                    X_train=X_train,
                    y_train=y_train,
                    timesteps=TIMESTEPS_PER_DATASET[dataset]
                )

                run_log.append({
                    'agent':      agent,
                    'dataset':    dataset,
                    'seed':       seed,
                    'model_path': model_path,
                    'curve_path': curve_path
                })

    log_path = os.path.join(BASE_DIR, 'results', 'run_log.csv')
    pd.DataFrame(run_log).to_csv(log_path, index=False)
    print(f"\n[OK] 27 runs completados. Log -> {log_path}")

    return run_log
