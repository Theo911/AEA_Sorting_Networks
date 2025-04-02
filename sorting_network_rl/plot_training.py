import matplotlib.pyplot as plt
import pandas as pd
import os
from config import MODEL_DIR, MODEL_NAME, N_WIRES, MAX_STEPS


def plot_log():
    run_id = f"{N_WIRES}wires_{MAX_STEPS}steps"
    run_dir = os.path.join(MODEL_DIR, run_id)
    log_path = os.path.join(run_dir, MODEL_NAME.replace(".pt", "_log.csv"))

    if not os.path.exists(log_path):
        print("Log file not found.")
        return

    df = pd.read_csv(log_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df['Episode'], df['TotalReward'], label='Total Reward', alpha=0.6)
    plt.plot(df['Episode'], df['AvgReward'], label='Average Reward (window)', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Reward Progression During Training ({N_WIRES} wires, {MAX_STEPS} steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_log()