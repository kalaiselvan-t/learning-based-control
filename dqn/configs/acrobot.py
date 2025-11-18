"""Acrobot-v1 DQN Configuration"""
import torch

CONFIG = {
    # Environment
    'env_name': 'Acrobot-v1',

    # Network architecture
    'hidden_dim': 128,

    # DQN hyperparameters
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'buffer_size': 50000,
    'batch_size': 64,
    'target_update_freq': 100,

    # Exploration strategy
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,

    # Training
    'episodes': 500,
    'max_steps': 500,

    # Warm-up
    'warmup_steps': 1000,

    # Early stopping
    'early_stop': True,
    'early_stop_threshold': -100,
    'early_stop_window': 100,

    # Persistence
    'save_path': 'models/acrobot.pth',
    'plot_path': 'plots/acrobot.png',

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
