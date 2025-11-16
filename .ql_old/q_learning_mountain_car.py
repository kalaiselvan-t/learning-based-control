import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import tqdm

def run(episodes: int, is_training: bool=True, render: bool=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    bin_size = 20
    learning_rate = 0.2
    discount_factor = 0.95

    epsilon_start = 0.5
    epsilon_min = 0.01
    epsilon_decay_power = 2  # Inverse exponential: slow decay early, fast at end
    rng = np.random.default_rng()

    pos_space = np.linspace(start = env.observation_space.low[0], stop = env.observation_space.high[0], num = bin_size)
    vel_space = np.linspace(start = env.observation_space.low[1], stop = env.observation_space.high[1], num = bin_size)

    if is_training:
        q_table = np.zeros(shape = (len(pos_space)+1, len(vel_space)+1, env.action_space.n))
    else:
        with open('mountain_car.pkl', 'rb') as f:
            q_table = pickle.load(f)
    
    rewards_per_episode = np.zeros(episodes)

    # Testing metrics
    success_count = 0
    episode_lengths = []

    for episode in tqdm(range(episodes)):

        state = env.reset()[0]

        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False
        truncated = False
        rewards = 0
        steps = 0

        # Calculate epsilon for this episode (inverse exponential decay)
        if is_training:
            epsilon = epsilon_min + (epsilon_start - epsilon_min) * (1 - episode / episodes) ** epsilon_decay_power
        else:
            epsilon = 0  # No exploration during testing

        while not (terminated or truncated):

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

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
            steps += 1

        # Track testing metrics
        if not is_training:
            if terminated:  # Reached goal (position >= 0.5)
                success_count += 1
            episode_lengths.append(steps)

        rewards_per_episode[episode] = rewards

        # Early stopping if environment is solved
        if is_training and episode >= 100:
            recent_avg = np.mean(rewards_per_episode[episode-100:episode+1])
            if recent_avg > -110:
                print(f"\n✓ Environment solved at episode {episode+1}!")
                print(f"  Average reward (last 100 episodes): {recent_avg:.2f}")
                rewards_per_episode = rewards_per_episode[:episode+1]
                break

    env.close()

    # Save trained model
    if is_training:
        with open('mountain_car.pkl', 'wb') as f:
            pickle.dump(q_table, f)
        print(f"\n✓ Model saved to mountain_car.pkl")

    # Print testing statistics
    if not is_training:
        print(f"\n{'='*50}")
        print(f"TESTING RESULTS")
        print(f"{'='*50}")
        print(f"Success rate: {success_count}/{len(episode_lengths)} ({100*success_count/len(episode_lengths):.1f}%)")
        print(f"Average reward: {np.mean(rewards_per_episode):.2f} ± {np.std(rewards_per_episode):.2f}")
        print(f"Average steps: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"{'='*50}\n")

    # Plot training progress
    if is_training:
        actual_episodes = len(rewards_per_episode)
        mean_rewards = np.zeros(actual_episodes)
        for t in range(actual_episodes):
            mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100) : (t+1)])

        plt.figure(figsize=(10, 6))
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (last 100 episodes)')
        plt.title('MountainCar-v0 Q-Learning Training Progress')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=-110, color='r', linestyle='--', label='Solved threshold (-110)')
        plt.legend()
        plt.savefig('mountain_car.png')
        print(f"✓ Training plot saved to mountain_car.png")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'train':
            # Training mode: 5k episodes, no rendering
            print("="*50)
            print("TRAINING MODE")
            print("="*50)
            print("Training for 5,000 episodes...")
            print("(Will stop early if solved)\n")
            run(5000, is_training=True, render=False)

        elif mode == 'test':
            # Testing mode: 100 episodes, no rendering, show stats
            print("="*50)
            print("TESTING MODE")
            print("="*50)
            print("Testing for 100 episodes...\n")
            run(100, is_training=False, render=False)

        elif mode == 'demo':
            # Demo mode: 5 episodes with visualization
            print("="*50)
            print("DEMO MODE")
            print("="*50)
            print("Running 5 episodes with visualization...\n")
            run(5, is_training=False, render=True)

        else:
            print("Usage: python q_learning_mountain_car.py [train|test|demo]")
            print("  train - Train agent for 10,000 episodes")
            print("  test  - Test trained agent for 100 episodes (no rendering)")
            print("  demo  - Run 5 episodes with visualization")
            sys.exit(1)

    else:
        # Default: demo mode
        print("="*50)
        print("DEMO MODE (default)")
        print("="*50)
        print("Running 5 episodes with visualization...")
        print("Use 'python q_learning_mountain_car.py train' to train first\n")
        run(5, is_training=False, render=True)
