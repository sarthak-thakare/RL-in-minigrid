import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
log_path = "storage/DoorKey/log.csv"  # Update if your file is elsewhere
try:
    df = pd.read_csv(log_path)
    print("Training log loaded successfully.")
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit()

# Plot 1: Return Mean, Std, Min, Max over Frames
if all(col in df.columns for col in ['return_mean', 'return_std', 'return_min', 'return_max']):
    fig1, ax1 = plt.subplots()
    ax1.plot(df['frames'].values, df['return_mean'].values, label='Return Mean', color='green')
    ax1.fill_between(df['frames'].values, df['return_min'].values, df['return_max'].values, color='lightgreen', alpha=0.5, label='Return Min/Max')
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Return")
    ax1.set_title("Return per Episode (Mean, Min, Max) over Frames")
    ax1.grid(True)
    ax1.legend()
    plt.show()

# Plot 2: Policy Loss & Value Loss over Frames
if 'policy_loss' in df.columns and 'value_loss' in df.columns:
    fig2, ax2 = plt.subplots()
    ax2.plot(df['frames'].values, df['policy_loss'].values, label='Policy Loss', color='red')
    ax2.plot(df['frames'].values, df['value_loss'].values, label='Value Loss', color='blue')
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Loss")
    ax2.set_title("Policy Loss & Value Loss over Frames")
    ax2.grid(True)
    ax2.legend()
    plt.show()

# Plot 3: Gradient Norm over Frames
if 'grad_norm' in df.columns:
    fig3, ax3 = plt.subplots()
    ax3.plot(df['frames'].values, df['grad_norm'].values, label='Gradient Norm', color='purple')
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("Gradient Norm")
    ax3.set_title("Gradient Norm over Frames")
    ax3.grid(True)
    ax3.legend()
    plt.show()
