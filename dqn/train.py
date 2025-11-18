import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt

from agent import DQNAgent

def train(config):
    """
    Train DQN agent on specified environment.

    Args:
        config: dict, configuration parameters
    """
    print("=" * 50)
    print(f"TRAINING: {config['env_name']}")
    print(f"Device: {config['device']}")
    print("=" * 50)

    # Create environment
    env = gym.make(config['env_name'])

    if config.get('use_balance_wrapper', False):
        if config['env_name'] == 'Acrobot-v1':
            from wrappers.acrobot_balance import AcrobotBalanceWrapper
            env = AcrobotBalanceWrapper(
                env,
                balance_duration=config['balance_duration'],
                angle_threshold=config['angle_threshold']
            )
        elif config['env_name'] == 'Pendulum-v1':
            from wrappers.pendulum_balance import PendulumBalanceWrapper
            env = PendulumBalanceWrapper(
                env,
                balance_duration=config['balance_duration'],
                angle_threshold=config['angle_threshold']
            )
        print(f"Using balance wrapper: {config['balance_duration']}s duration, {config['angle_threshold']} rad threshold")

    if config.get('use_action_discretizer', False):
        from wrappers.action_discretizer import ActionDiscretizer
        env = ActionDiscretizer(
            env,
            n_bins=config['n_action_bins']
        )
        print(f"Using action discretizer: {config['n_action_bins']} discrete actions")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Episodes: {config['episodes']}")
    print()

    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq'],
        device=config['device']
    )

    # Training storage
    rewards_per_episode = []
    losses = []
    epsilon = config['epsilon_start']

    # Warm-up: fill buffer with random experiences
    print(f"Warm-up: collecting {config['warmup_steps']} random experiences...")
    state, _ = env.reset()
    for _ in range(config['warmup_steps']):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)

        state = next_state if not done else env.reset()[0]
    
    print(f"Buffer size: {len(agent.replay_buffer)}\n")

    # Training loop
    for episode in tqdm(range(config['episodes']), desc="Training"):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = []
        steps = 0

        done = False
        while not done and steps < config['max_steps']:
            # Select action
            action = agent.select_action(state, epsilon)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transistion
            agent.store_transition(state, action, reward, next_state, done)

            # Learn
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
        
        # Decay epsilon
        epsilon = max(config['epsilon_min'], epsilon * config['epsilon_decay'])

        # Record metrics
        rewards_per_episode.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Early stopping check
        if config['early_stop'] and episode >= config['early_stop_window']:
            recent_avg = np.mean(rewards_per_episode[-config['early_stop_window']:])
            if recent_avg >= config['early_stop_threshold']:
                print(f"Environement solved at episode {episode + 1}!")
                print(f"Average reward (last {config['early_stop_window']} episodes): {recent_avg:.2f}")
                break
    
    env.close()

    # Save model
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    agent.save(config['save_path'])
    print(f"\n✓ Model saved to {config['save_path']}")

    # Plot results
    plot_training(rewards_per_episode, losses, config)

    return agent, rewards_per_episode

def plot_training(rewards_per_episode, losses, config):
    """
    Plot training progress and save to file.
    
    Args:
        rewards_per_episode: list, rewards for each episode
        losses: list, average loss per episode
        config: dict, configuration parameters
    """
    episodes = len(rewards_per_episode)

    # Calculate moving average
    window = min(100, episodes)
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - window + 1):(t + 1)])

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot rewards
    ax1.plot(mean_rewards, linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel(f'Average Reward (last {window} episodes)', fontsize=12)
    ax1.set_title(f"{config['env_name']} DQN Training Progress", fontsize=14)
    ax1.grid(True, alpha=0.3)

    if 'early_stop_threshold' in config:
        ax1.axhline(
            y=config['early_stop_threshold'],
            color='r',
            linestyle='--',
            label=f"Solved threshold ({config['early_stop_threshold']})"
        )
        ax1.legend()

    # Plot losses
    if losses:
        ax2.plot(losses, linewidth=2, color='orange')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Average Loss', fontsize=12)
        ax2.set_title('Training Loss', fontsize=14)
        ax2.grid(True, alpha=0.3)

    # Save plot
    os.makedirs(os.path.dirname(config['plot_path']), exist_ok=True)
    plt.tight_layout()
    plt.savefig(config['plot_path'], dpi=150, bbox_inches='tight')
    print(f"Training plot saved to {config['plot_path']}")
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train.py [acrobot|cartpole|pendulum|pendulum_balance]")
        print("\nAvailable environments:")
        print("  acrobot          - Acrobot-v1 (balance task)")
        print("  cartpole         - CartPole-v1")
        print("  pendulum         - Pendulum-v1 (swing-up)")
        print("  pendulum_balance - Pendulum-v1 (balance task)")
        sys.exit(1)

    env_choice = sys.argv[1].lower()

    # Load config
    if env_choice == 'acrobot':
        from configs.acrobot import CONFIG
    elif env_choice == 'cartpole':
        from configs.cartpole import CONFIG
    elif env_choice == 'pendulum':
        from configs.pendulum import CONFIG
    elif env_choice == 'pendulum_balance':
        from configs.pendulum_balance import CONFIG
    else:
        print(f"Unknown environment: {env_choice}")
        sys.exit(1)
    
    # Train
    agent, rewards = train(CONFIG)

    print("\n Training complete!")