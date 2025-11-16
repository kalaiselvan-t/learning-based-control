""" CartPole-v1 Q-Learning Configuration"""

CONFIG = {
    # Environment
    'env_name': 'CartPole-v1',

    # Discretization (need custom bins for cartpole)
    'bins': [6, 6, 12, 12],         # [cart_pos, cart_vel, pole_angle, pole_vel]

    # Custom bounds (Cartpole has inf velocity bounds)
    'custom_bounds': {
        0: (-2.4, 2.4),     # cart position (from env)
        1: (-3.0, 3.0),     # cart velocity (clipped)
        2: (-0.5, 0.5),     # pole angle (radians, approx 28 deg)
        3: (-2.0, 2.0),     # pole velocity (clipped)
    },

    # Q-learning hyperparameters
    'learning_rate': 1.0,
    'discount_factor': 0.99,    # Higher for longer episodes

    # Exploration strategy
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 'linear',

    # Training
    'episodes': 10000,
    'max_steps': 500,

    # Early Stopping
    'early_stop': True,
    'early_stop_threshold': 195,    # CartPole solved threshold
    'early_stop_window': 100,

    # Persistance
    'save_path': 'models/cartpole.pkl',
    'plot_path': 'plots/cartpole.png',
}