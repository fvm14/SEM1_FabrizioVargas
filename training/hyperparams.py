PPO_PARAMS = {
    'learning_rate':  3e-4,
    'n_steps':        2048,
    'n_epochs':       10,
    'batch_size':     64,
    'clip_range':     0.2,
    'gamma':          0.99,
    'gae_lambda':     0.95,
    'ent_coef':       0.0,
    'vf_coef':        0.5,
    'max_grad_norm':  0.5,
    'policy_kwargs':  {'net_arch': [64, 64]}
}

DQN_PARAMS = {
    'learning_rate':           1e-4,
    'buffer_size':             100_000,
    'batch_size':              32,
    'target_update_interval':  500,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps':   0.05,
    'exploration_fraction':    0.1,
    'gamma':                   0.99,
    'train_freq':              4,
    'gradient_steps':          1,
    'policy_kwargs':           {'net_arch': [64, 64]}
}

DDQN_PARAMS = {
    **DQN_PARAMS,
}

TOTAL_TIMESTEPS = 2_000_000
TIMESTEPS_PER_DATASET = {
    'nslkdd':     2_000_000,
    'cicids2017': 2_000_000,
    'unswnb15':   2_000_000
}

CICIDS_SUBSAMPLE_SIZE = 125_000
SEEDS    = [42, 123, 2026]
DATASETS = ['nslkdd', 'cicids2017', 'unswnb15']
AGENTS   = ['PPO', 'DQN', 'DDQN']
