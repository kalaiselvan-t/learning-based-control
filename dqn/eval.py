import sys
import torch
import numpy as np
import gymnasium as gym

from agent import DQNAgent

def evaluate(config, num_episodes=10, render=True):
    """
    Evaluate trained DQN agent.
    
    Args:
        config: dict, configuration parameters
        num_episodes: int, number of test episodes
        render: bool, whether to render environment
    """
    print("=" * 50)
    print(f"EVALUATING: {config['env_name']}")
    print(f"Device: {config['device']}")
    print("=" * 50)

    # Create environment
    render_mode = 'human' if render else None
    env = gym.make(config['env_name'], render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

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

    # Load trained model
    try:
        agent.load(config['save_path'])
        print(f"✓ Loaded model from {config['save_path']}\n")
    except FileNotFoundError:
        print(f"✗ Model not found at {config['save_path']}")
        print("  Train the agent first using: python train.py [env_name]")
        sys.exit(1)

    # Evaluation loop
    total_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < config['max_steps']:
            # Greedy action (no exploration)
            action = agent.select_action(state, epsilon=0.0)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Count success (environment-specific)
        if config['env_name'] == 'CartPole-v1' and total_reward >= 195:
            successes += 1
        elif config['env_name'] == 'Acrobot-v1' and terminated:
            successes += 1

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

    env.close()

    # Print statistics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {successes}/{num_episodes} ({100 * successes / num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Best reward: {max(total_rewards):.2f}")
    print(f"Worst reward: {min(total_rewards):.2f}")
    print("=" * 50)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python eval.py [acrobot|cartpole] [--no-render]")
        print("\nOptions:")
        print("  --no-render  Disable visualization (faster)")
        sys.exit(1)

    env_choice = sys.argv[1].lower()
    render = '--no-render' not in sys.argv

    # Load config
    if env_choice == 'acrobot':
        from configs.acrobot import CONFIG
    elif env_choice == 'cartpole':
        from configs.cartpole import CONFIG
    else:
        print(f"Unknown environment: {env_choice}")
        sys.exit(1)

    # Evaluate
    evaluate(CONFIG, num_episodes=10, render=render)