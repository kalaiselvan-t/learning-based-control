'''
- Rewards are +1 for every incremental timestep
- Termination:
    - If the pole falls over too far
    - If the cart moves more than 2.4 units from the center
- Better performing scenarios will run for longer duration, accumulating larger return
'''
# imports
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(filename='dqn.log', filemode='w', level=logging.INFO)

LOG = True
# ------------------------------------------------------------------
# setup the environment
env = gym.make("CartPole-v1", render_mode = "human")
obs, info = env.reset()
env.render()
if LOG: logger.info('Env set up')
if LOG: logger.info('Obs: %s', obs)

# ------------------------------------------------------------------
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# ------------------------------------------------------------------
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if LOG: logger.info('Device: %s', device)
# ------------------------------------------------------------------

# ==================================================================
#                             Replay Memory
# ==================================================================

# Transistion 
#   - represents a single transistion in our env
#   - maps (state, action) to (next_state, reward)
Transistion = namedtuple('Transistion', 
                         ('state', 'action', 'next_state', 'reward'))

# Replay memory
#   - a cyclic buffer of bounded size that holds the transistions observed recently
#   - also implements a .sample() method for selecting a random batch of transistions

class ReplayMemory(object):                                         # 'object' passed in as arg is optional. it is default in python3. see subclassing 

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transistion(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
# ------------------------------------------------------------------

# ==================================================================
#                               DQN Algorithm
# ==================================================================

# Q-network
class DQN(nn.Module):
    '''
    Input: 4 states of the environment
    Output: 2 actions that the agent can take
    '''
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# ------------------------------------------------------------------
# ==================================================================
#                               Training
# ==================================================================
# Hyperparameters

BATCH_SIZE = 128    # number of transistions sampled from the replay buffer
GAMMA = 0.99        # disocount factor

# Epsilon - the probability with which a random action is chosen
EPS_START = 0.9     # starting value of epsilon
EPS_DECAY = 1000    # controls the rate of exponential decay of epsilon. higher means a slower decay
EPS_END = 0.05      # final value of epsilon


TAU = 0.005     # update rate of the target network
LR = 1e-4       # learning rate of the AdamW optimizer

# ------------------------------------------------------------------
n_actions = env.action_space.n

state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
# a = A()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
# ------------------------------------------------------------------

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []
# ------------------------------------------------------------------

def plot_durations(show_result = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# ==================================================================
#                          Training Loop
# ==================================================================

def optimize_model():

    if len(memory) < BATCH_SIZE:
        return
    
    transistions = memory.sample(BATCH_SIZE)
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
    
    state_batch = torch.cat(batch.state) #128 * state
    if LOG: logger.info('state_batch:\n %s', state_batch)

    action_batch = torch.cat(batch.action)
    if LOG: logger.info('action_batch:\n %s', action_batch)

    reward_batch = torch.cat(batch.reward)
    if LOG: logger.info('reward_batch:\n %s', reward_batch)

    # selects the q-values for the actions we took
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    if LOG: logger.info('state_action_values:\n %s', state_action_values)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if LOG: logger.info('next_state_values:initialized:\n %s', next_state_values)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    if LOG: logger.info('next_state_values:updated:\n %s', next_state_values)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    if LOG: logger.info('expected_state_action_values:\n %s', expected_state_action_values)

    # compute huber loss
    criterion = nn.SmoothL1Loss()
    if LOG: logger.info('criterion:\n %s', criterion)

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    if LOG: logger.info('loss:\n %s', loss)

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    if LOG: logger.info('--------------------------------------------------')

# ------------------------------------------------------------------
# main training loop

if torch.cuda.is_available() or torch.backends.mps.is_available():
    # num_episodes = 600
    num_episodes = 10
else:
    # num_episodes = 50
    num_episodes = 10

for i_episode in range(num_episodes):
    # initialize the env to get the state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)    # [-0.004 -0.010 0.037 -0.034] -> [[-0.004, -0.010,  0.037, -0.034]]
    
    for t in count():                                                               # count() -> creates a sequence of numbers infinitely. acts as a counter here
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # store the transistion in memory
        memory.push(state, action, next_state, reward)

        # move to the next state
        state = next_state

        # perform one step of the optimization
        optimize_model()

        # soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key] * (1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t+1)
            plot_durations()
            break

print('complete')

if LOG: logger.info('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
