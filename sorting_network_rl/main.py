import numpy as np
import os
import torch
import csv
from env.sorting_env import SortingNetworkEnv
from agent.dqn_agent import DQNAgent
from utils.state_encoder import encode_state
from utils.evaluation import count_sorted_inputs
from config import *
import time


def train():
    # Setup environment and agent
    env = SortingNetworkEnv(n_wires=N_WIRES, max_steps=MAX_STEPS)
    state_dim = encode_state(N_WIRES, MAX_STEPS, []).shape[0]
    action_dim = env.get_action_space_size()
    agent = DQNAgent(state_dim, action_dim, lr=LEARNING_RATE, gamma=GAMMA,
                     epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                     epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

    # Update model directory based on current config
    run_id = f"{N_WIRES}wires_{MAX_STEPS}steps"
    run_dir = os.path.join(MODEL_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, MODEL_NAME)
    epsilon_path = model_path.replace(".pt", "_epsilon.npy")
    log_path = os.path.join(run_dir, MODEL_NAME.replace(".pt", "_log.csv"))

    # Load model and epsilon if checkpoint exists
    if os.path.exists(model_path):
        try:
            agent.policy_net.load_state_dict(torch.load(model_path))
            agent.update_target_network()
            if os.path.exists(epsilon_path):
                agent.epsilon = float(np.load(epsilon_path))
            print(f"Model and epsilon loaded. Resuming training from saved checkpoint.")
        except RuntimeError as e:
            print(f"\n⚠️ Model incompatibil: {e}\nÎncepem antrenamentul de la zero.\n")

    # Prepare logging
    reward_history = []
    with open(log_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Episode", "TotalReward", "AvgReward", "Epsilon"])

        for episode in range(1, NUM_EPISODES + 1):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state_vector = encode_state(N_WIRES, MAX_STEPS, state['comparators'])
                action = agent.select_action(torch.FloatTensor(state_vector).unsqueeze(0))

                next_state, _, done = env.step(action)
                next_state_vector = encode_state(N_WIRES, MAX_STEPS, next_state['comparators'])

                correct = count_sorted_inputs(N_WIRES, next_state['comparators'])
                reward = correct / (2 ** N_WIRES)

                agent.store_transition(state_vector, action, reward, next_state_vector, done)
                agent.train_step()
                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            reward_history.append(total_reward)

            if episode % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            if episode % PRINT_EVERY == 0:
                avg_reward = np.mean(reward_history[-PRINT_EVERY:])
                print(f"Episode {episode}: Total reward = {total_reward:.2f}, Avg reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
                writer.writerow([episode, total_reward, avg_reward, agent.epsilon])
                log_file.flush()

                torch.save(agent.policy_net.state_dict(), model_path)
                np.save(epsilon_path, np.array(agent.epsilon))
                print(f"Checkpoint saved at episode {episode}.")

    print("Training complete.")


if __name__ == "__main__":
    train()