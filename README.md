# 🧠 Reinforcement Learning with MiniGrid

This project provides a Streamlit-based interactive interface for experimenting with Reinforcement Learning (RL) techniques on the MiniGrid environment. It supports manual control, Q-learning, PPO training, performance comparison, and visualization — all from your browser!

---

## 🚀 Features

### 1. Manual Run
- Use **arrow keys** to control the agent manually.
- Press **space bar** to open doors.
- Great for understanding environment dynamics hands-on.

### 2. Q-Learning
- Run an agent trained using **Q-learning**.
- Observe how it behaves and learns from the grid environment.

### 3. Compare Q-Learning vs A* Performance
- Analyze and **compare performance** metrics between the learned Q-values and classical **A\*** pathfinding algorithm.

### 4. Boltzmann Exploration
- Implemented in a **dynamic obstacle** environment.
- Allows the agent to explore probabilistically based on state-action values.

---

## 📄 PPO Training Tab

Train an agent using **Proximal Policy Optimization (PPO)** by selecting custom arguments:

- `env`: For example, `MiniGrid-Empty-5x5-v0`
- `model`: Model name to save, e.g., `Empty`
- `interval`: Evaluation interval, e.g., `10`
- `frames`: Number of training frames, e.g., `80000`

---

## 📊 Evaluation Tab

- Load and evaluate your trained PPO model.
- Provides summary statistics and success rate analysis.

---

## 🎮 Visualization Tab

- Select environment and trained model.
- Click **Visualize** to watch the agent interact with the environment.
- Press `Ctrl+C` in terminal to exit Streamlit server.

---

## ✅ Steps to Run

1. Install dependencies:
   - Streamlit
   - MiniGrid
   - Gym
   - (Others depending on your setup)

2. In terminal, run:
   ```bash
   streamlit run Main_ui.py
