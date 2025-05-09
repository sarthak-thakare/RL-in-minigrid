import minigrid
import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

env = gym.make("MiniGrid-Empty-8x8-v0")

# Hyperparameters
alpha = 0.1           # Learning rate: How much new information overrides old information
gamma = 0.99          # Discount factor: How much future rewards matter
epsilon = 1.0         # Exploration factor: ε-greedy policy
min_epsilon = 0.1     # Minimum epsilon (stopping exploration)
epsilon_decay = 0.995 # How much epsilon decays after each episode
num_episodes = 600    # Number of training episodes

# Q-table: state (tuple) -> action values
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Extract custom state from environment
def get_state(env):
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir

    goal_pos = None
    for i in range(env.unwrapped.grid.width):
        for j in range(env.unwrapped.grid.height):
            obj = env.unwrapped.grid.get(i, j)
            if obj and obj.type == "goal":
                goal_pos = (i, j)
                break
        if goal_pos:
            break

    return agent_pos + goal_pos + (agent_dir,)

# Tracking performance
rewards_per_episode = []
successes = 0

# Training loop
for episode in range(1, num_episodes + 1):
    obs, info = env.reset()
    state = get_state(env)
    done = False
    total_reward = 0

    while not done:
        # ε-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = get_state(env)
        done = terminated or truncated

        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)
    if reward > 0:
        successes += 1

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 50 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.2f} | ε: {epsilon:.3f} | Success rate: {successes/episode:.2f}")

env.close()

# Plot results
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning on MiniGrid-Empty-8x8-v0")
plt.grid(True)
plt.savefig("e-greedy_loss.png")
plt.show()

# Final test run with rendering
test_env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
obs, info = test_env.reset()
state = get_state(test_env)
done = False

while not done:
    action = np.argmax(Q[state])
    obs, reward, terminated, truncated, _ = test_env.step(action)
    state = get_state(test_env)
    done = terminated or truncated

test_env.close()
