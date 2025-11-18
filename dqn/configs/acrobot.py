"""Acrobot-v1 DQN Configuration"""
import torch

CONFIG = {
    # Environment
    'env_name': 'Acrobot-v1',

    # Network architecture
    'hidden_dim': 128,          # Neurons per hidden layer

    # DQN hyperparameters
    'learning_rate': 1e-3,      # Adam optimizer learning rate
    'gamma': 0.99,              # Discount factor
    'buffer_size': 50000,       # Replay buffer capacity
    'batch_size': 64,           # Samples per training batch
    'target_update_freq': 100,  # Steps between target network syncs

    # Exploration strategy
    'epsilon_start': 1.0,       # Initial exploration
    'epsilon_min': 0.01,        # Minimum exploration
    'epsilon_decay': 0.995,     # Exponential decay per episode

    # Training
    'episodes': 500,            # Training episodes
    'max_steps': 500,           # Max steps per episode (Acrobot default)

    # Warm-up (Collect experiences before training)
    'warmup_steps': 1000,       # Random actions to fill buffer

    # Early stopping
    'early_stop': True,
    'early_stop_threshold': -100,   # Acrobot solved at -100 avg reward
    'early_stop_window': 100,

    # Persistence
    'save_path': 'models/acrobot.pth',
    'plot_path': 'plots/acrobot.png',

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}