"""Pendulum-v1 Balance Task DQN Configuration"""
import torch

CONFIG = {
    # Environment
    'env_name': 'Pendulum-v1',
    'use_action_discretizer': True,
    'n_action_bins': 21,                # Fine torque control
    'use_balance_wrapper': True,        # Use custom balance task
    'balance_duration': 2.0,
    'angle_threshold': 0.02,            # 0.1 radians (~5.7 degrees)

    # Curriculum
    'use_curriculum': True,             # Progressively alter threshold according to schedule
    'curriculum_schedule': [
        (0, 0.10),          # Episodes 0-199: start easy with 0.1rad
        (200, 0.08),
        (400, 0.06),
        (600, 0.05),
        (800, 0.04),
        (1000, 0.03),
        (1200, 0.02),
    ],

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
    'episodes': 2000,
    'max_steps': 200,

    # Warm-up
    'warmup_steps': 2000,

    # Early stopping
    'early_stop': False,
    'early_stop_threshold': 200,
    'early_stop_window': 100,

    # Persistence
    'save_path': 'models/pendulum_balance.pth',
    'plot_path': 'plots/pendulum_balance.png',

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
