import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
from numpy import save

env_name = "MiniGrid-Dynamic-Obstacles-Random-6x6-v0"

# --- Boltzmann Exploration function ---
def take_action_boltzmann(q_value, temperature):
    exp_q = np.exp(q_value / temperature)
    probabilities = exp_q / np.sum(exp_q)
    return np.random.choice(len(q_value), p=probabilities)

# --- Initialize Environment ---
env = gym.make(env_name, render_mode=None)
env.reset()
inner_env = env.unwrapped

# --- Q-learning parameters ---
q_value = np.zeros((3, 16, 4, 2))  # (actions, position_idx, direction, obstacle)
episodes = 1000
gamma = 0.9
alpha = 0.3
temperature = 1.0

total_reward = np.zeros(episodes)
avg_loss_boltzmann = np.zeros(episodes)

# --- Training Loop using Boltzmann Exploration ---
for K in range(episodes):
    print("Episode (Boltzmann):", K + 1)
    env.reset()
    inner_env = env.unwrapped

    temperature = max(temperature - 0.002, 0.05)  # Decay temperature

    terminated = False
    truncated = False

    front_cell = inner_env.grid.get(*inner_env.front_pos)
    not_clear = front_cell and front_cell.type != "goal"
    x3 = 1 if not_clear else 0
    x1 = (inner_env.agent_pos[0] - 1) * 4 + (inner_env.agent_pos[1] - 1)
    x2 = inner_env.agent_dir

    action = take_action_boltzmann(q_value[:, x1, x2, x3], temperature)
    losses = []

    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward[K] += reward

        new_x1 = (inner_env.agent_pos[0] - 1) * 4 + (inner_env.agent_pos[1] - 1)
        new_x2 = inner_env.agent_dir
        front_cell = inner_env.grid.get(*inner_env.front_pos)
        not_clear = front_cell and front_cell.type != "goal"
        new_x3 = 1 if not_clear else 0

        new_action = take_action_boltzmann(q_value[:, new_x1, new_x2, new_x3], temperature)

        td_target = reward + gamma * q_value[new_action, new_x1, new_x2, new_x3]
        td_error = td_target - q_value[action, x1, x2, x3]
        q_value[action, x1, x2, x3] += alpha * td_error

        losses.append(td_error ** 2)

        x1, x2, x3 = new_x1, new_x2, new_x3
        action = new_action

    avg_loss_boltzmann[K] = np.mean(losses) if losses else 0
    print(f"Total reward: {total_reward[K]}, Avg Loss: {avg_loss_boltzmann[K]}")

# --- Save Trained Values ---
save("boltzmann_reward.npy", total_reward)
save("boltzmann_q_values.npy", q_value)
save("boltzmann_loss.npy", avg_loss_boltzmann)
env.close()

# --- Plot and Save Loss Convergence Graph ---
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, episodes + 1), avg_loss_boltzmann, label='Boltzmann Loss (MSE)', color='green')
plt.xlabel("Episode")
plt.ylabel("Average TD Error")
plt.title("Loss Convergence: Boltzmann Exploration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("boltzmann_loss.png")
plt.show()

# --- Demo with Rendering ---
print("\nTraining Complete! Running demo...\n")
demo_env = gym.make(env_name, render_mode="human")

episod_itter = 10
iterr = 0

while (iterr < episod_itter):
    seed = np.random.randint(0, 10000)
    demo_env.reset(seed=seed)
    inner_env = demo_env.unwrapped

    terminated = False
    truncated = False

    front_cell = inner_env.grid.get(*inner_env.front_pos)
    not_clear = front_cell and front_cell.type != "goal"
    x3 = 1 if not_clear else 0
    x1 = (inner_env.agent_pos[0] - 1) * 4 + (inner_env.agent_pos[1] - 1)
    x2 = inner_env.agent_dir

    while not (terminated or truncated):
        action = np.argmax(q_value[:, x1, x2, x3])
        _, _, terminated, truncated, _ = demo_env.step(action)

        x1 = (inner_env.agent_pos[0] - 1) * 4 + (inner_env.agent_pos[1] - 1)
        x2 = inner_env.agent_dir
        front_cell = inner_env.grid.get(*inner_env.front_pos)
        not_clear = front_cell and front_cell.type != "goal"
        x3 = 1 if not_clear else 0

    print(f"Demo episode done. Restarting with seed {seed}...\n")
    iterr += 1
