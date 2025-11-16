import pickle   # Converts python objects to byte stream and back
import numpy as np

class QLearningAgent:
    """
    Core Q-learning algorithm

    Q-learning update rule:
    Q(s,a) <- Q(s,a) +  α[r + y max_a' Q(s', a') - Q(s,a)]

    where,
     α(alpha) -> learning rate
     Y(gamma) -> discount_factor
     s = current state, a = current action
     s' = next state, r = reward
    """

    def __init__(self, state_shape: tuple, n_actions: int, learning_rate: float, discount_factor: float):
        """
        Args:
            state_shape: dim of discretizes state space
            n_actions: no of possible actions
            learning_rate: step size for Q-updates (alpha)
            discount_factor: importance of future rewards (Y)
        """
        # Initialize q-learning table with zeros
        self.q_table: np.ndarray = np.zeros(
            shape=(*state_shape, n_actions)
            )

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.n_actions = n_actions
    
    def select_action(self, state: tuple, epsilon: float):
        """
        Epsilon-greedy action selection
        
        Args:
            state: discretized state indices
            epsilon: probability of random action [0,1]
        """
        # With probability epsilon, choose random action (exploration)
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)     # Choose a random number between [0, n_actions]
        else:
            return np.argmax(self.q_table[state])   # Choose max action from q-values of all actions for that state
    
    def update(self, state, action, reward, next_state):
        """Q-learning update"""
        # Q-value we currently believe for taking action a in state s. the old estimate we are about to update
        current_q = self.q_table[state][action]

        # Best Q-value for next state (greedy, regardless of what we actually did)
        max_next_q = np.max(self.q_table[next_state])

        # TD target: r + y * max Q(s', a')
        td_target = reward + self.discount_factor * max_next_q

        # TD error: target - current
        td_error = td_target - current_q

        # Q-learning update: Q(s,a) += a * TD_error
        self.q_table[state][action] += self.learning_rate * td_error

    
    def save(self, filepath):
        """Save Q-table to disk"""
        # Pickle the Q-table
        with open(filepath, 'wb') as f:     # wb - write binary
            pickle.dump(self.q_table, f)
    
    def load(self, filepath):
        """Load Q-table from disk"""
        # Load pickled Q-table
        with open(filepath, 'rb') as f:     # rb - read binary
            self.q_table = pickle.load(f)
