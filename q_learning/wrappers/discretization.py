import numpy as np
import gymnasium as gym

class DiscretizationWrapper(gym.ObservationWrapper):
    """
    Wraps continuous observation space into discrete bins.

    Example:
        Position: [-1.2, 0.6] → 20 bins → indices [0, 19]
        Velocity: [-0.07, 0.07] → 20 bins → indices [0, 19]
    """

    def __init__(self, env, bins, custom_bounds=None):
        """
        Args:
            env: Gym environment
            bins: int or list of ints
                  - int: same bins for all dimensions (e.g., 20)
                  - list: different bins per dimension (e.g., [20, 10])
            custom_bounds: dict, optional custom bounds for dimensions with inf
                  - {dim_index: (low, high)}
                  - e.g., {1: (-3.0, 3.0), 3: (-2.0, 2.0)} for CartPole velocities
        """
        super().__init__(env)

        # Get observation space dimensionality
        n_dims = len(self.env.observation_space.low)

        # Handle both int and list inputs
        if isinstance(bins, int):
            self.bins = [bins] * n_dims  # Same bins for all dims
        else:
            self.bins = bins  # Different bins per dim

        # Store custom bounds (for handling inf bounds)
        self.custom_bounds = custom_bounds or {}

        self.bin_spaces = self._create_bins()

    def _create_bins(self):
        """
        Create discretization bins for each dimension.

        Returns:
            List of arrays, each containing bin edges for one dimension
        """
        bin_spaces = []

        # Iterate over each observation dimension
        for i, n_bins in enumerate(self.bins):
            # Check if custom bounds provided for this dimension
            if i in self.custom_bounds:
                low, high = self.custom_bounds[i]   # Use custom
            else:
                low = self.env.observation_space.low[i]     # Use env
                high = self.env.observation_space.high[i]

            # Create bins from low to high
            bins = np.linspace(start=low, stop=high, num=n_bins)
            bin_spaces.append(bins)

        return bin_spaces

    def observation(self, obs):
        """
        Convert continuous observation to discrete state.

        Args:
            obs: array, continuous observation (e.g., [-0.52, 0.01])

        Returns:
            tuple of ints, discretized state indices (e.g., (10, 5))
        """
        # Digitize each dimension separately
        discrete_state = []
        for i, value in enumerate(obs):
            idx = np.digitize(value, self.bin_spaces[i])
            discrete_state.append(idx)

        return tuple(discrete_state)  # Tuple for Q-table indexing
