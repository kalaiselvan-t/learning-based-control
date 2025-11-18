import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for DQN

    Stores transistions (s, a, r, s', done) and samples random batches.
    Uses deque for efficient FIFO when buffer is full
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: int, maximum number of transistions to store
        """
        self.buffer = deque(maxlen=capacity)    # Autoremoves oldest when full

    def push(self, state, action, reward, next_state, done):
        """
        Add a transistion to the buffer

        Args:
            state: array, current state observation
            action: int, action taken
            reward: float, reward received
            next_state: array, next state observed
            done: bool, whether episode terminated
        """
        # Store as tuple
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transistions.

        Args:
            batch_size: int, number of transistions to sample

        Returns:
            tuple of batched arrays: (states, actions, rewards, next_states, dones)
        """
        # Random sample
        batch = random.sample(self.buffer, batch_size)

        # Unzip into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)      # zip(*batch) transposes list of tuples into tuples of lists

        # Convert to numpy arrays for easier batching
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)     # 0 or 1
        )
    
    def __len__(self):
        """
        Return current buffer size.
        """
        return len(self.buffer)


