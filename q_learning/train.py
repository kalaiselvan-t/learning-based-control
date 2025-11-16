import os
import sys
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt

from agent import QLearningAgent
from wrappers.discretization import DiscretizationWrapper

def get_epsilon(episode, config):
    """
    Calculate epsilon for current episode based on decay strategy.

    Args: 
        episode: int, current episode number
        config: dict, configuration with epsilon parameters
    
    Returns:
        float, epsilon value for this episode
    """
    eps_start = config['epsilon_start']
    eps_min = config['epsilon_min']
    episodes = config['episodes']

    if config['epsilon_decay'] == 'linear':
        # Linear decay: eps = start - (start - min) * (episode / total)
        epsilon = eps_start - (eps_start - eps_min) * (episode / episodes)
    elif config['epsilon_decay'] == 'power':
        # Power decay: eps = min + (start - min) * (1 - progress)^power
        progress = episode / episodes
        power = config.get('epsilon_power', 2)
        epsilon = eps_min + (eps_start - eps_min) * (1 - progress) ** power
    else:
        raise ValueError(f"Unknown epsilon decay: {config['epsilon_decay']}")
    
    return max(epsilon, eps_min)    # Ensure never below minimum

def train(config):
    """
    Train Q-learning agent on specified environment

    Args:
        config: dict, configuration parameters
    """

    print("=" * 50)
    print(f"Training: {config['env_name']}")
    print("=" * 50)

    # Create environment
    env = gym.make(config['env_name'])

    # Wrap with discretization
    custom_bounds = config.get('custom_bounds', None)
    env = DiscretizationWrapper(env, bins=config['bins'], custom_bounds=custom_bounds)

    # Get state space shape for Q-table
    # Need to add +1 to each bin count (np.digitize can return bin_count)
    if isinstance(config['bins'], int):
        n_dims = len(env.env.observation_space.low)
        state_shape = tuple([config['bins'] + 1] * n_dims)
    else:
        state_shape = tuple([b + 1 for b in config['bins']])
    
    n_actions = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Actions: {n_actions}")
    print(f"Episodes: {config['episodes']}")
    print()

    # Create agent
    agent = QLearningAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor']
    )

    # Training storage
    rewards_per_episode = []

    # Training loop
    for episode in tqdm(range(config['episodes']), desc="Training"):
        state, _ = env.reset()
        epsilon = get_epsilon(episode, config)

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < config['max_steps']:
            # Select action
            action = agent.select_action(state, epsilon)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-table
            agent.update(state, action, reward, next_state)

            # Move to next state
            state = next_state
            total_reward += reward
            steps += 1
        rewards_per_episode.append(total_reward)

        # Early stopping check
        if config['early_stop'] and episode >= config['early_stop_window']:
            recent_avg = np.mean(rewards_per_episode[-config['early_stop_window']:])
            if recent_avg >= config['early_stop_threshold']:
                print(f"\n  Environment solved at episode {episode + 1}!")
                print(f"    Average reward (last {config['early_stop_window']} episodes): {recent_avg:.2f}")
                break
        
    env.close()

    # Save model
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    agent.save(config['save_path'])
    print(f"\n✓ Model saved to {config['save_path']}")

    # Plot results
    plot_training(rewards_per_episode, config)

    return agent, rewards_per_episode

def plot_training(rewards_per_episode, config):
    """
    Plot training progress and save to file.
    
    Args:
        rewards_per_episode: list, rewards for each episode
        config: dict, configuration parameters
    """
    episodes = len(rewards_per_episode)

    # Calculate moving average (window = 100)
    window = min(100, episodes)
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - window + 1):(t + 1)])

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Average Reward (last {window} episodes)', fontsize=12)
    plt.title(f"{config['env_name']} Q-Learning Training Progress", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add solved threshold line
    if 'early_stop_threshold' in config:
        plt.axhline(
            y=config['early_stop_threshold'],
            color='r',
            linestyle='--',
            label=f"Solved threshold ({config['early_stop_threshold']})"
        )
        plt.legend()

    # Save plot
    os.makedirs(os.path.dirname(config['plot_path']), exist_ok=True)
    plt.savefig(config['plot_path'], dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved to {config['plot_path']}")
    plt.close()

if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python train.py [mountain_car|cartpole]")
        print("\nAvailable environments:")
        print("  mountain_car - MountainCar-v0")
        print("  cartpole     - CartPole-v1")
        sys.exit(1)
    
    env_choice = sys.argv[1].lower()

    # Load config
    if env_choice == 'mountain_car':
        from configs.mountain_car import CONFIG
    elif env_choice == 'cartpole':
        from configs.cartpole import CONFIG
    else:
        print(f"Unknown environment: {env_choice}")
        print("Choose from: mountain_car, cartpole")
        sys.exit(1)
    
    # Train
    agent, rewards = train(CONFIG)
 
