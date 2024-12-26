import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

env = gym.make("MountainCar-v0", render_mode="rgb_array")

DISCRETE_BIN_SIZE = 20
LEARNING_RATE = 0.1
DISCOUNT = 0.95             # how important is future actions over current actions | current forward or future reward
EPISODES = 25000
GOAL_POSITION = 0.5
SHOW_EVERY = 2000

obs_space_range = env.observation_space.high - env.observation_space.low
win_size = (obs_space_range) / DISCRETE_BIN_SIZE
q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_BIN_SIZE, DISCRETE_BIN_SIZE, env.action_space.n))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / win_size
    return tuple(discrete_state.astype(np.int64))

for episode in tqdm(range(EPISODES)):

    obs, info = env.reset()
    discrete_state = get_discrete_state(obs)
    done = False

    frames = []
    
    while not done:
        action = np.argmax(q_table[discrete_state])

        next_state, reward, terminated, truncated, info = env.step(action)
        next_discrete_state = get_discrete_state(next_state)

        if episode % SHOW_EVERY == 0:
            frames.append(env.render())

        done = terminated or truncated

        if not done:
            future_q_value = np.max(q_table[next_discrete_state])
            cur_q_value = q_table[discrete_state + (action, )]
            new_q_val = (1 - LEARNING_RATE) * cur_q_value + LEARNING_RATE * (reward + DISCOUNT * future_q_value)

            q_table[discrete_state + (action, )] = new_q_val

        elif next_state[0] >= GOAL_POSITION:
            print(f"goal reached at {episode}")
            q_table[discrete_state + (action, )] = 0
    
    gif_name = str(episode) + ".gif"
    if len(frames) > 0:  imageio.mimsave(gif_name, frames, fps=30)

env.close()

