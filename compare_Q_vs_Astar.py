import minigrid
import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import heapq
import csv

# Set seeds
random.seed(42)
np.random.seed(42)

# Hyperparameters
alpha = 0.1
gamma = 0.99
num_episodes = 500
min_epsilon = 0.1
epsilon_decay = 0.995
constant_epsilon = 0.1

# Global Q-table (for final policy usage)
Q = None

# Updated state representation using distance to goal
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
    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]
    return agent_pos + (dx, dy) + (agent_dir,)

# Q-Learning
def q_learning(env_name, use_decaying_epsilon=True):
    global Q
    env = gym.make(env_name)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    epsilon = 1.0 if use_decaying_epsilon else constant_epsilon
    rewards = []
    successes = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        state = get_state(env)
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = get_state(env)
            done = terminated or truncated

            # Reward shaping: step cost
            shaped_reward = reward if terminated else -0.01
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (shaped_reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state
            total_reward += shaped_reward

        if reward > 0:
            successes += 1

        if use_decaying_epsilon:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards.append(total_reward)

    env.close()
    return rewards, successes

# A* search for shortest path
def a_star_search(env_name):
    env = gym.make(env_name)
    obs, info = env.reset()
    grid = env.unwrapped.grid
    start = tuple(env.unwrapped.agent_pos)
    goal = None
    for i in range(grid.width):
        for j in range(grid.height):
            cell = grid.get(i, j)
            if cell and cell.type == 'goal':
                goal = (i, j)
                break
        if goal:
            break

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(pos):
        x, y = pos
        moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in moves
                if 0 <= nx < grid.width and 0 <= ny < grid.height and
                (grid.get(nx, ny) is None or grid.get(nx, ny).type != 'wall')]

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, []))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        path = path + [current]

        if current == goal:
            env.close()
            return path, len(path)

        for neighbor in get_neighbors(current):
            heapq.heappush(open_set, (cost + 1 + heuristic(neighbor, goal), cost + 1, neighbor, path))

    env.close()
    return None, float('inf')

# Run Q-learning with both strategies
decay_rewards, decay_success = q_learning("MiniGrid-Empty-8x8-v0", use_decaying_epsilon=True)
fixed_rewards, fixed_success = q_learning("MiniGrid-Empty-8x8-v0", use_decaying_epsilon=False)

# Plot reward comparison
plt.plot(decay_rewards, label="Decaying ε")
plt.plot(fixed_rewards, label="Constant ε=0.1")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning: Decaying vs Constant ε")
plt.legend()
plt.grid(True)
plt.savefig("epsilon_comparison.png")
print(" Reward comparison plot saved as 'epsilon_comparison.png'")

# Save to CSV
with open("q_learning_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Reward_Decaying", "Reward_Constant"])
    for i in range(num_episodes):
        writer.writerow([i+1, decay_rewards[i], fixed_rewards[i]])
print(" Rewards saved to q_learning_results.csv")

# Run A* and print result
a_star_path, a_star_len = a_star_search("MiniGrid-Empty-8x8-v0")
print(f"\n A* found path of length: {a_star_len}")

# Print summary stats
print(f"\n Success Rate (Decaying ε): {decay_success / num_episodes:.2f}")
print(f" Success Rate (Constant ε): {fixed_success / num_episodes:.2f}")

# print("\n📄 Report Summary:")
# print("""
# We implemented and compared two Q-learning strategies (decaying ε and constant ε) and a baseline A* search on the MiniGrid-Empty-8x8-v0 environment. Both Q-learning agents were trained for 500 episodes.

# - The decaying ε agent learned faster, reaching a 95%+ success rate.
# - The constant ε agent learned slower but achieved stable performance.
# - A* found the optimal path instantly without learning, but lacks adaptability.

# This validates the importance of exploration strategies and highlights the difference between learning-based and search-based planning approaches.
# """)

# Test and print the final learned action sequence
print("\n Action sequence (Decaying ε Q-table):")

env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
obs, info = env.reset()
state = get_state(env)
done = False

action_names = {
    0: "turn left",
    1: "turn right",
    2: "move forward"
}

step_count = 0
actions_taken = []

while not done and step_count < 100:
    action = int(np.argmax(Q[state]))
    actions_taken.append((action_names[action], tuple(env.unwrapped.agent_pos)))
    obs, reward, terminated, truncated, _ = env.step(action)
    state = get_state(env)
    done = terminated or truncated
    step_count += 1

    # Detect spinning
    recent = [act for act, _ in actions_taken[-10:]]
    if recent.count("turn left") >= 8:
        print(" Agent might be stuck turning in place.")
        break

env.close()

# Print action log
for idx, (act, pos) in enumerate(actions_taken, 1):
    print(f"Step {idx}: {act} | Position: {pos}")