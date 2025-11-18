import numpy as np
import gymnasium as gym

class AcrobotBalanceWrapper(gym.Wrapper):
    """
    Modify Acrobot to require balancing upright for N seconds.

    Default Acrobot: Episode ends when tip swings above target height once.
    This wrapper: Must balance near upright position for duration.
    """
    def __init__(self, env, balance_duration=5.0, angle_threshold=0.2):
        """
        Args:
            env: Acrobot-v1 environment
            balance_duration: float, seconds to balance (default 5.0)
            angle_threshold: float, max angle deviation from upright (radians)
        """
        super().__init__(env)

        # Convert duration to steps (Acrobot runs at 50 FPS by default)
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

        # Check if acrobot is upright
        # obs = [cos(theta), sin(theta), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
        # Upright when theta1 = pi and theta2 = 0
        theta1 = np.arctan2(obs[1], obs[0])     # Extract theta1 from cos/sin
        theta2 = np.arctan2(obs[3], obs[2])     # Extract theta2

        # Check if close to upright (theta1 = pi, theta2 = 0)
        # Convert theta1 to deviation from pi
        theta1_dev = abs(theta1 - np.pi) if theta1 > 0 else abs(theta1 + np.pi)
        theta2_dev = abs(theta2)

        is_balanced = (theta1_dev < self.angle_threshold and
                        theta2_dev < self.angle_threshold)
        
        # Update balanced steps counter
        if is_balanced:
            self.balanced_steps += 1
        else:
            self.balanced_steps = 0     # Reset if falls
        
        # Custom reward - gradient toward upright
        height_reward = -(theta1_dev + theta2_dev)  # Range: [-2π, 0]

        if is_balanced:
            reward = 10.0  # High reward for balancing
        else:
            reward = height_reward  # Gradient toward upright

        # Bonus for consecutive balanced steps
        reward += self.balanced_steps * 0.1

        # Success termination: balanced for required duration
        if self.balanced_steps >= self.balance_steps_required:
            terminated = True
            reward = 100.0      # Bonus for success
            info['success'] = True

        # Removed strict failure termination - let it learn gradually
        
        info['balanced_steps'] = self.balanced_steps
        info['is_balanced'] = is_balanced

        return obs, reward, terminated, truncated, info
