import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from network import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network agent.

    Uses two networks (online and target) with experience replay
    to learn Q-values via temporal difference learning
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        device='cuda'
    ):
        """
        Args:
            state_dim: int, dimension of state space
            action_dim: int, number of discrete actions
            learning_rate: float, optimizer learning rate
            gamma: float, discount factor for future rewards
            buffer_size: int, replay buffer capacity
            batch_size: int, number of samples per training batch
            target_update_freq: int, steps between target network updates
            device: str, 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Networks
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())       # Copy weights
        self.target_net.eval()      # Target network in eval mode (no training)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training counter
        self.learn_step_counter = 0
    
    def select_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.

        Args:
            state: array, current state
            epsilon: float, exploration probability

        Returns:
            action: int, selected action
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploit: greedy action from Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)    # (1, state_dim)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)    # (1, action_dim)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self):
        """
        Sample batch from replay buffer and update online network.

        Returns:
            loss: float, TD loss value (for logging)
        """
        # check if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)              # (batch, state_dim)
        actions = torch.LongTensor(actions).to(self.device)             # (batch,)
        rewards = torch.FloatTensor(rewards).to(self.device)            # (batch,)
        next_states = torch.FloatTensor(next_states).to(self.device)    # (batch, state_dim)
        dones = torch.FloatTensor(dones).to(self.device)                # (batch,)

        # Compute current Q-values: Q(s,a)
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # gather(1, actions) selects Q-value for taken action
        # Shape: (batch, action_dim) -> (batch, 1)

        # Computer target Q-values: r + y * max Q_target(s')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]     # Max Q-value for next state
            target_q = rewards + self.gamma * next_q * (1 - dones)  # Zero out if episode is done
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        return loss.item()
    
    def save(self, filepath):
        """
        Save online network weights.
        """
        torch.save(self.online_net.state_dict(), filepath)
    
    def load(self, filepath):
        """
        Load network weights.
        """
        self.online_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())


    