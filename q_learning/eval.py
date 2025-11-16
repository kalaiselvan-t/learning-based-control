import gymnasium as gym
import sys
from agent import QLearningAgent
from wrappers.discretization import DiscretizationWrapper


def evaluate(config, num_episodes=10, render=True):
    """
    Evaluate trained Q-learning agent.
    
    Args:
        config: dict, configuration parameters
        num_episodes: int, number of test episodes
        render: bool, whether to render environment
    """
    print("=" * 50)
    print(f"EVALUATING: {config['env_name']}")
    print("=" * 50)

    # Create environment
    render_mode = 'human' if render else None
    env = gym.make(config['env_name'], render_mode=render_mode)

    # Wrap with discretization
    custom_bounds = config.get('custom_bounds', None)
    env = DiscretizationWrapper(env, bins=config['bins'], custom_bounds=custom_bounds)

    # Get state space shape
    if isinstance(config['bins'], int):
        n_dims = len(env.env.observation_space.low)
        state_shape = tuple([config['bins'] + 1] * n_dims)
    else:
        state_shape = tuple([b + 1 for b in config['bins']])

    n_actions = env.action_space.n

    # Create agent and load trained model
    agent = QLearningAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=config['learning_rate'],  # Not used in eval
        discount_factor=config['discount_factor']
    )

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

        if terminated:  # Reached goal
            successes += 1

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

    env.close()

    # Print statistics
    import numpy as np
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {successes}/{num_episodes} ({100 * successes / num_episodes:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("=" * 50)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python eval.py [mountain_car|cartpole] [--no-render]")
        print("\nOptions:")
        print("  --no-render  Disable visualization (faster)")
        sys.exit(1)

    env_choice = sys.argv[1].lower()
    render = '--no-render' not in sys.argv

    # Load config
    if env_choice == 'mountain_car':
        from configs.mountain_car import CONFIG
    elif env_choice == 'cartpole':
        from configs.cartpole import CONFIG
    else:
        print(f"Unknown environment: {env_choice}")
        sys.exit(1)

    # Evaluate
    evaluate(CONFIG, num_episodes=10, render=render)