import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transistion(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DoublePendulum():

    def __init__(self,episodes: int , training: bool = False):
        self.is_training = training

        if self.is_training:
            self.episodes = 1
        else:
            self.episodes = episodes
        
        print(f"episodes: {self.episodes}")

        self.setup_env()

        self.n_actions = 1
        self.n_observations = len(self.state)

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []

        print(f"action: {self.select_action(self.state)}")
        self.training()

    def setup_env(self):
        self.env = gym.make('InvertedDoublePendulum-v5', render_mode='human' if not self.is_training else None)
        self.state, _ = self.env.reset()
        if LOG:
            logging.info(f"state: {self.state} \n")
            logging.info(f"shape: {self.state.shape}")
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(self.env.action_space.sample(), device=device, dtype=torch.float32)
    
    def plot_durations(self, show_result = False):
        plt.figure(1)
        #print(f"episodes: {self.episodes}")
        durations_t = torch.tensor((self.episode_durations), dtype=torch.float)
        #print(f"durations: {durations_t}, duration-shape: {durations_t.shape}")
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        # Take 100 episode averages and plot them
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001) # pausing to update the plots
    
    def optimize_model(self):

        if len(self.memory) < BATCH_SIZE:
            return
        
        transistions = self.memory.sample(BATCH_SIZE)
        if LOG: logger.info('==============================================')
        if LOG: logger.info('transistions:\n %s', transistions)
        # converts batch-array of transistions to Transistion of batch-arrays
        '''
        Eg: 
            [Transisiton(state='state3', action=3), Transisiton(state='state4', action=4), Transisiton(state='state2', action=2)]
                                            to
            Transisiton(state=('state3', 'state4', 'state2'), action=(3, 4, 2))
        '''
        batch = Transistion(*zip(*transistions))        
        if LOG: logger.info('batch:\n %s', batch)
        '''
        returns a mask 
        Eg: (True, True, True)
        '''
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        if LOG: logger.info('non_final_mask:\n %s', non_final_mask)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None])
        if LOG: logger.info('non_final_next_states:\n %s', non_final_next_states)    
        
        #state_batch = torch.cat(batch.state) #128 * state
        state_batch = torch.cat([torch.tensor(state, dtype=torch.float32) for state in batch.state]).to(device)
        print("state batch shape", state_batch.shape)
        if LOG: logger.info('state_batch:\n %s', state_batch)

        #action_batch = torch.cat(batch.action)
        action_batch = torch.cat([torch.tensor(action, dtype=torch.long) for action in batch.action]).to(device)
        print("action batch shape", action_batch.shape)
        if LOG: logger.info('action_batch:\n %s', action_batch)

        #reward_batch = torch.cat(batch.reward)
        reward_batch = torch.cat([torch.tensor(reward, dtype=torch.float32) for reward in batch.reward]).to(device)
        print("reward batch shape", reward_batch.shape)
        if LOG: logger.info('reward_batch:\n %s', reward_batch)

        # selects the q-values for the actions we took
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        state_action_values = self.policy_net(state_batch)
        if LOG: logger.info('state_action_values:\n %s', state_action_values)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if LOG: logger.info('next_state_values:initialized:\n %s', next_state_values)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        if LOG: logger.info('next_state_values:updated:\n %s', next_state_values)

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        if LOG: logger.info('expected_state_action_values:\n %s', expected_state_action_values)

        # compute huber loss
        criterion = nn.SmoothL1Loss()
        if LOG: logger.info('criterion:\n %s', criterion)

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        if LOG: logger.info('loss:\n %s', loss)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if LOG: logger.info('--------------------------------------------------')

    def training(self):
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            # num_episodes = 600
            num_episodes = 600
        else:
            # num_episodes = 50
            num_episodes = 10

        for i_episode in tqdm(range(num_episodes)):
            # initialize the env to get the state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)    # [-0.004 -0.010 0.037 -0.034] -> [[-0.004, -0.010,  0.037, -0.034]]
            
            for t in count():                                                               # count() -> creates a sequence of numbers infinitely. acts as a counter here
                action = self.select_action(state)
                print(f"action: {action}, shape: {action.shape}")
                action = action.cpu()
                action = action.numpy()
                print(f"action: {action}")
                if not action.shape == (1,):
                    print(f"action_shape: {action.shape}")
                    action = action.reshape((1,))
                observation, reward, terminated, truncated, _ = self.env.step(action)
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # store the transistion in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # perform one step of the optimization
                self.optimize_model()

                # soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key] * (1-TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t+1)
                    self.plot_durations()
                    break

    def __del__(self):
        logging.info("===================================\ncomplete")
        self.env.close()

class DQN(nn.Module):
    '''
    Input: 4 states of the environment
    Output: 2 actions that the agent can take
    '''
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)

        self.tanh = nn.Tanh()
    
    def forward(self, x):
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.tanh(x)

Transistion = namedtuple('Transistion', 
                                      ('state', 'action', 'next_state', 'reward'))

logger = logging.getLogger(__name__)
logging.basicConfig(filename='double-pendulum/dp.log', filemode='w', level=logging.INFO)

LOG = True

plt.ion()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if LOG: logging.info(f"Device: {device} \n")

BATCH_SIZE = 128
GAMMA = 0.99

EPS_START = 0.9
EPS_DECAY = 1000
EPS_END = 0.05

TAU = 0.005
LR = 1e-4

agent = DoublePendulum(episodes=100)