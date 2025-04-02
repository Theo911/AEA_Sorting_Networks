import os
import torch
from env.sorting_env import SortingNetworkEnv
from agent.dqn_agent import DQNAgent
from utils.state_encoder import encode_state
from utils.evaluation import is_sorting_network
from config import *


def evaluate_agent():
    run_id = f"{N_WIRES}wires_{MAX_STEPS}steps"
    run_dir = os.path.join(MODEL_DIR, run_id)
    model_path = os.path.join(run_dir, MODEL_NAME)

    env = SortingNetworkEnv(n_wires=N_WIRES, max_steps=MAX_STEPS)
    state_dim = encode_state(N_WIRES, MAX_STEPS, []).shape[0]
    action_dim = env.get_action_space_size()
    agent = DQNAgent(state_dim, action_dim)

    if not os.path.exists(model_path):
        print("Model not found.")
        return

    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    state = env.reset()
    done = False
    while not done:
        state_vector = encode_state(N_WIRES, MAX_STEPS, state['comparators'])
        action = agent.select_action(torch.FloatTensor(state_vector).unsqueeze(0))
        state, _, done = env.step(action)

    comparators = state['comparators']
    print("\nEvaluated Comparator Network:")
    for idx, (i, j) in enumerate(comparators):
        print(f"Step {idx+1}: compare wire {i} with wire {j}")

    if is_sorting_network(N_WIRES, comparators):
        print("\n✅ Valid sorting network!")
    else:
        print("\n❌ Not a valid sorting network.")

    return comparators

def visualize_network(comparators):
    grid = [[" "] * len(comparators) for _ in range(N_WIRES)]
    for t, (i, j) in enumerate(comparators):
        for wire in range(N_WIRES):
            if wire == i or wire == j:
                grid[wire][t] = "●"
            else:
                grid[wire][t] = "─"

    print("\nVisual representation:")
    for wire in range(N_WIRES):
        print(f"w{wire}: " + " ".join(grid[wire]))

if __name__ == "__main__":
    comparators = evaluate_agent()
    if comparators:
        visualize_network(comparators)