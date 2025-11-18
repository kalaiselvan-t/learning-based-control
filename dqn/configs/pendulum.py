"""Pendulum-v1 DQN Configuration"""
import torch

CONFIG = {
    # Environment
    'env_name': 'Pendulum-v1',
    'use_action_discretizer': True,
    'n_action_bins': 21,

    # Network architecture
    'hidden_dim': 256,

    # DQN hyperparameters
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'buffer_size': 100000,
    'batch_size': 128, 
    'target_update_freq': 200, 

    # Exploration strategy
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,

    # Training
    'episodes': 1000,
    'max_steps': 200,               # Pendulum default episode length

    # Warm-up
    'warmup_steps': 2000,

    # Early stopping
    'early_stop': True,
    'early_stop_threshold': -200,   # Good performance for Pendulum
    'early_stop_window': 100,

    # Persistence
    'save_path': 'models/pendulum.pth',
    'plot_path': 'plots/pendulum.png',

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
