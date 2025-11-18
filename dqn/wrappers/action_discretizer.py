import numpy as np
import gymnasium as gym

class ActionDiscretizer(gym.ActionWrapper):
    """
    Discretizes continuous action spaces for use with DQN.

    Converts continuous action space into discrete bins.
    """
    def __init__(self, env, n_bins=11):
        """
        Args:
            env: Environment with continuous action space
            n_bins: Number of discrete actions to create
        """
        super().__init__(env)

        # Store original action space bounds
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]
        self.n_bins = n_bins

        # Create discrete action space
        self.action_space = gym.spaces.Discrete(n_bins)

        # Precompute discrete action values
        self.discrete_actions = np.linspace(self.low, self.high, n_bins)

    def action(self, action):
        """
        Convert discrete action index to continuous action value.

        Args:
            action: int, discrete action index

        Returns:
            continuous action value as numpy array
        """
        continuous_action = self.discrete_actions[action]
        return np.array([continuous_action], dtype=np.float32)
