"""Pendulum-v1 Balance Task DQN Configuration"""
import torch

CONFIG = {
    # Environment
    'env_name': 'Pendulum-v1',
    'use_action_discretizer': True,
    'n_action_bins': 21,  # Fine torque control
    'use_balance_wrapper': True,  # Use custom balance task
    'balance_duration': 1.0,  # 1 second balance (same as Acrobot)
    'angle_threshold': 0.1,   # 0.1 radians (~5.7 degrees) - same as Acrobot

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
    'epsilon_decay': 0.998,

    # Training
    'episodes': 1000,
    'max_steps': 200,

    # Warm-up
    'warmup_steps': 2000,

    # Early stopping
    'early_stop': True,
    'early_stop_threshold': 50,  # Similar to Acrobot
    'early_stop_window': 100,

    # Persistence
    'save_path': 'models/pendulum_balance.pth',
    'plot_path': 'plots/pendulum_balance.png',

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
