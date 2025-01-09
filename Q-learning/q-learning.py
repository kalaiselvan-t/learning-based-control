import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import logging

LOG = True

# setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='q-learning.log', filemode='w', level=logging.INFO)

def run(env_name: str, episodes: int, is_training: bool, render: bool):
    # setup env
    env = gym.make(env_name, render_mode='human' if render else None)
    obs, info = env.reset()
    if LOG: logger.info('Obs: %s \n', obs)

    # hyperparameters
    learning_rate = 0.9
    discount_factor = 0.9

    epsilon = 1.0
    epsilon_decay = 2 / episodes
    if LOG: logger.info('Epsilon decay: %s \n', epsilon_decay)

    bin_size = 20
    rng = np.random.default_rng()

    # discretize space
    cart_state_space = np.linspace(start=env.observation_space.low[0], stop=env.observation_space.high[0], num=bin_size)
    cart_vel_space = np.linspace(start=env.observation_space.low[1], stop=env.observation_space.high[1], num=bin_size)
    pendulum_state_space = np.linspace(start = env.observation_space.low[2], stop = env.observation_space.high[2], num = bin_size)
    pendulum_vel_space = np.linspace(start=env.observation_space.low[3], stop=env.observation_space.high[3], num=bin_size)

    if LOG: 
        logger.info('Cart state space: %s \n', cart_state_space)
        logger.info('Shape: %s \n', cart_state_space.shape)

        logger.info('Cart vel space: %s \n', cart_vel_space)
        logger.info('Shape: %s \n', cart_vel_space.shape)

        logger.info('Pole state space: %s \n', pendulum_state_space)
        logger.info('Shape: %s \n', pendulum_state_space.shape)

        logger.info('Pole vel space: %s \n', pendulum_vel_space)
        logger.info('Shape: %s \n', pendulum_vel_space.shape)

    # initialize q-table and other storage variables
    if is_training:
        #q_table = np.zeros(shape=(len(cart_state_space), len(cart_vel_space), len(pendulum_state_space), len(pendulum_vel_space), env.action_space.n))
        q_table = np.zeros(shape = (len(cart_state_space)+1, len(cart_vel_space)+1, len(pendulum_state_space)+1, len(pendulum_vel_space)+1, env.action_space.n))
    else:
        f = open('q_table.pkl', 'rb')
        q_table = pickle.load(f)
        f.close()
    
    if LOG: 
        logger.info('q-table: %s \n', q_table)
        logger.info('Shape: %s \n', q_table.shape)


    rewards_per_episode = np.zeros(episodes)

    # training loop
    for episode in tqdm(range(episodes)):
        # reset env at the begining of a new episode
        obs, info = env.reset()

        # bookkeeping
        done = False
        rewards = 0
        
        # discretize the states
        dis_cart_pos = np.digitize(obs[0], cart_state_space)
        dis_cart_vel = np.digitize(obs[1], cart_vel_space)
        dis_pole_pos = np.digitize(obs[2], pendulum_state_space)
        dis_pole_vel = np.digitize(obs[3], pendulum_vel_space)

        if LOG: 
            logger.info('Dis cart pos: %s \n', dis_cart_pos)
            logger.info('Shape: %s \n', dis_cart_pos.shape)

            logger.info('Dis cart vel: %s \n', dis_cart_vel)
            logger.info('Shape: %s \n', dis_cart_vel.shape)

            logger.info('Dis pole pos: %s \n', dis_pole_pos)
            logger.info('Shape: %s \n', dis_pole_pos.shape)

            logger.info('Dis pole vel: %s \n', dis_pole_vel)
            logger.info('Shape: %s \n', dis_pole_vel.shape)

        eps_length = 0

        while not done:
            # choose action based on epsilon-greedy
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[dis_cart_pos, dis_cart_vel, dis_pole_pos, dis_pole_vel, :])

            if LOG: logger.info('action: %s , shape: %s \n', action, action.shape)

            # perform a step
            next_obs, reward, terminated, truncated, info = env.step(action)

            if LOG:
                logger.info('Reward: %s \n', reward)
                logger.info('Terminated: %s \n', terminated)
                logger.info('Truncated: %s \n', truncated)

            # discretise next state
            next_dis_cart_pos = np.digitize(next_obs[0], cart_state_space)
            next_dis_cart_vel = np.digitize(next_obs[1], cart_vel_space)
            next_dis_pole_pos = np.digitize(next_obs[2], pendulum_state_space)
            next_dis_pole_vel = np.digitize(next_obs[3], pendulum_vel_space)

            if LOG: 
                logger.info('Dis next cart pos: %s \n', next_dis_cart_pos)
                logger.info('Shape: %s \n', next_dis_cart_pos.shape)

                logger.info('Dis next cart vel: %s \n', next_dis_cart_vel)
                logger.info('Shape: %s \n', next_dis_cart_vel.shape)

                logger.info('Dis next pole pos: %s \n', next_dis_pole_pos)
                logger.info('Shape: %s \n', dis_pole_pos.shape)

                logger.info('Dis next pole vel: %s \n', next_dis_pole_vel)
                logger.info('Shape: %s \n', next_dis_pole_vel.shape)

            # check if the episode is terminated
            done = terminated or truncated
            if not is_training:
                print(f"Terminated: {terminated}, Truncated: {truncated}")

            # update q-table during training
            if is_training:
                update_value = learning_rate * (
                        reward + discount_factor * np.max(q_table[next_dis_cart_pos, next_dis_cart_vel, next_dis_pole_pos, next_dis_pole_vel,:]) - q_table[dis_cart_pos, dis_cart_vel, dis_pole_pos, dis_pole_vel, action]            
                )
            
                if LOG:
                    logger.info('Update value: %s \n', update_value)
                    logger.info('q_table value: %s \n', q_table[dis_cart_pos, dis_cart_vel, dis_pole_pos, dis_pole_vel, action])
                
                q_table[dis_cart_pos, dis_cart_vel, dis_pole_pos, dis_pole_vel, action] = q_table[dis_cart_pos, dis_cart_vel, dis_pole_pos, dis_pole_vel, action] + update_value
            
            # move to next state
            dis_cart_pos = next_dis_cart_pos
            dis_cart_vel = next_dis_cart_vel
            dis_pole_pos = next_dis_pole_pos
            dis_pole_vel = next_dis_pole_vel

            rewards += reward
            eps_length += 1
        
        if not is_training:
            print(f"Episode length: {eps_length}")

        # decay epsilon
        epsilon = max(epsilon - epsilon_decay, 0)

        # bookeeping
        rewards_per_episode[episode] = rewards
    
    # close environment
    env.close()

    # save the q-table to use for inference
    if is_training:
        f = open('q_table.pkl', 'wb')
        pickle.dump(q_table, f)
        f.close()

    mean_rewards = np.zeros(episodes)

    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100) : (t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'q_learning.png')


if __name__ == '__main__':
    if LOG: logger.info('Program start\n')
    #run('CartPole-v1', 17500, True, False)
    run('CartPole-v1', 10, False, True)