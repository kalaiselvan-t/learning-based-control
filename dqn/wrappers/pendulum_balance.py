import numpy as np
import gymnasium as gym

class PendulumBalanceWrapper(gym.Wrapper):
    """
    Modify Pendulum to require balancing upright for N seconds.

    Default Pendulum: Continuous reward based on angle and velocity.
    This wrapper: Must balance near upright position for duration.
    """
    def __init__(self, env, balance_duration=5.0, angle_threshold=0.2):
        """
        Args:
            env: Pendulum-v1 environment
            balance_duration: float, seconds to balance (default 5.0)
            angle_threshold: float, max angle deviation from upright (radians)
        """
        super().__init__(env)

        # Convert duration to steps (Pendulum runs at 50 FPS by default)
        self.balance_steps_required = int(balance_duration * 50)
        self.angle_threshold = angle_threshold

        # Track consecutive balanced steps
        self.balanced_steps = 0
        self.total_steps = 0

    def reset(self, **kwargs):
        """Reset environment and counters."""
        self.balanced_steps = 0
        self.total_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Modified step with balance-based rewards and termination.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        # Extract angle from observation
        # obs = [cos(theta), sin(theta), angular_velocity]
        # Upright when theta = 0
        theta = np.arctan2(obs[1], obs[0])  # Extract theta from cos/sin

        # Check if close to upright (theta = 0)
        theta_dev = abs(theta)

        is_balanced = theta_dev < self.angle_threshold

        # Update balanced steps counter
        if is_balanced:
            self.balanced_steps += 1
        else:
            self.balanced_steps = 0  # Reset if falls

        # Custom reward - gradient toward upright
        angle_reward = -theta_dev  # Range: [-π, 0]
        velocity_penalty = -0.1 * abs(obs[2])  # Penalize high velocity

        if is_balanced:
            # Precision bonus: reward being closer to perfect vertical
            # precision = 1.0 when theta_dev = 0, precision = 0.0 at threshold
            precision = 1.0 - (theta_dev / self.angle_threshold)
            reward = 10.0 + precision * 20.0  # Range: [10, 30]
            reward += velocity_penalty  # Still penalize high velocity
        else:
            reward = angle_reward + velocity_penalty

        # Bonus for consecutive balanced steps
        reward += self.balanced_steps * 0.1

        # Success termination: balanced for required duration
        if self.balanced_steps >= self.balance_steps_required:
            terminated = True
            reward = 100.0  # Bonus for success
            info['success'] = True

        info['balanced_steps'] = self.balanced_steps
        info['is_balanced'] = is_balanced

        return obs, reward, terminated, truncated, info
