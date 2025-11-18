import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Q-network for DQN

    Simple feedforward network (ML Perceptron) that takes states and outputs Q-values
    for each action
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Args:
            state_dim: int, dimension of state observation
            action_dim: int, dimension of discrete actions
            hidden_dim: int, number of neurons in hidden layers
        """
        super(QNetwork, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: tensor, shape(batch_size, state_dim) or (state_dim,)
        
        Returns:
            q_values: tensor, shape (batch_size, action_dim) or (action_dim,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)      # No activation on output (raw Q-values)
        return q_values
    