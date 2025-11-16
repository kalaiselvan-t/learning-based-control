"""MountainCar-v0 Q-learning Configuration"""

CONFIG = {
    # Environment
    'env_name': 'MountainCar-v0',

    # Discretization
    'bins': 20,  # Same bins for position and velocity
    # Or use:
    # 'bins': [20, 20]  # for explicit control

    # Q-learning hyperparameters
    'learning_rate': 0.2,        # α - step size 
    'discount_factor': 0.95,     # γ - future reward importance

    # Exploration strategy
    'epsilon_start': 0.5,       # Initial exploration
    'epsilon_min': 0.01,        # Minimum exploration
    'epsilon_decay' : 'power',  # Decay type: 'linear' or 'power'
    'epsilon_power': 2,         # For power decay

    # Training
    'episodes': 5000,
    'max_steps': 200,           # MountainCar default

    # Early stopping
    'early_stop': True,
    'early_stop_threshold': -110,   # Average reward target
    'early_stop_window': 100,       # Episodes to average

    # Persistence
    'save_path': 'models/mountain_car.pkl',
    'plot_path': 'plots/mountain_car.png',
}