import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def run(episodes: int, is_training: bool=True, render: bool=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    bin_size = 20
    learning_rate = 0.9
    discount_factor = 0.9

    epsilon = 1
    epsilon_decay_rate = 2 / episodes
    rng = np.random.default_rng()

    pos_space = np.linspace(start = env.observation_space.low[0], stop = env.observation_space.high[0], num = bin_size)
    vel_space = np.linspace(start = env.observation_space.low[1], stop = env.observation_space.high[1], num = bin_size)

    if is_training:
        q_table = np.zeros(shape = (len(pos_space), len(vel_space), env.action_space.n))
    else:
        f = open('Q-learning/mountain_car.pkl', 'rb')
        q_table = pickle.load(f)
        f.close()
    
    rewards_per_episode = np.zeros(episodes)

    for episode in tqdm(range(episodes)):

        state = env.reset()[0]

        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False
        rewards = 0

        while (not terminated and rewards > -900):

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, :])

            new_state, reward, terminated,_,_ = env.step(action)

            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q_table[state_p, state_v, action] = q_table[state_p, state_v, action] + learning_rate * (
                    reward + discount_factor * np.max(q_table[new_state_p, new_state_v,:]) - q_table[state_p, state_v, action]
                )
    
            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[episode] = rewards

    env.close()
    
    if is_training:
        f = open('mountain_car.pkl', 'wb')
        pickle.dump(q_table, f)
        f.close()

    mean_rewards = np.zeros(episodes)

    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100) : (t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')

if __name__ == '__main__':
    # run(5000, is_training=True, render=False)

    run(5, is_training=False, render=True)