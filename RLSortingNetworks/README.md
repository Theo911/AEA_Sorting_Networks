# RLSortingNetworks: Finding Sorting Networks using Deep Q-Learning

This project implements a Deep Reinforcement Learning (DRL) agent, specifically using Deep Q-Networks (DQN), to autonomously discover size-efficient sorting networks. The agent learns through interaction with a simulated environment, progressively adding comparators to build a network that can sort any input sequence for a given number of wires (`n`). The primary optimization objective is minimizing the total number of comparators (network size).

## Table of Contents

- [Overview](#overview)
  - [Problem Formulation](#problem-formulation-as-an-mdp)
  - [Approach: Deep Q-Learning](#approach-deep-q-learning)
- [Features](#features)
- [Project Structure](#project-structure)
- [Configuration (`configs/default_config.yaml`)](#configuration-configsdefault_configyaml)
- [How It Works: Implementation Details](#how-it-works-implementation-details)
  - [1. The Environment (`sorting_network_rl.env.sorting_env`)](#1-the-environment-sorting_network_rlenvsorting_env)
  - [2. State Representation (`sorting_network_rl.utils.state_encoder`)](#2-state-representation-sorting_network_rlutilsstate_encoder)
  - [3. The Agent (`sorting_network_rl.agent.dqn_agent`)](#3-the-agent-sorting_network_rlagentdqn_agent)
  - [4. The Q-Network Model (`sorting_network_rl.model.network`)](#4-the-q-network-model-sorting_network_rlmodelnetwork)
  - [5. The Training Process (`sorting_network_rl.core.trainer`)](#5-the-training-process-sorting_network_rlcoretrainer)
  - [6. Evaluation & Validation (`sorting_network_rl.utils.evaluation`)](#6-evaluation--validation-sorting_network_rlutilsevaluation)
  - [7. Pruning (`sorting_network_rl.utils.evaluation`)](#7-pruning-sorting_network_rlutilsevaluation)
  - [8. Visualization (`sorting_network_rl.utils.evaluation`)](#8-visualization-sorting_network_rlutilsevaluation)
  - [9. Evaluation Script (`sorting_network_rl.core.evaluator`, `scripts/evaluate.py`)](#9-evaluation-script-sorting_network_rlcoreevaluator-scriptsevaluatepy)
  - [10. Utilities (`sorting_network_rl.utils`)](#10-utilities-sorting_network_rlutils)
- [Usage](#usage)
  - [Training the Agent (`scripts/train.py`)](#training-the-agent-scriptstrainpy)
  - [Evaluating a Trained Agent (`scripts/evaluate.py`)](#evaluating-a-trained-agent-scriptsevaluatepy)
- [Results and Checkpoints](#results-and-checkpoints)
- [Future Work](#future-work)

## Overview

Sorting networks are comparator-based algorithms with a fixed structure, designed to sort sequences of inputs. They are particularly relevant for hardware implementations and parallel computing. Finding the *minimal* sorting network (in terms of the number of comparators) for a given number of inputs `n` is a challenging combinatorial problem.

This project tackles this challenge using Deep Reinforcement Learning. Instead of relying on explicit construction algorithms (like Batcher's sort), we train an agent to *learn* a policy for constructing these networks step-by-step.

### Problem Formulation (as an MDP)

We model the network construction process as a Markov Decision Process:

1.  **States:** The current partial sorting network, represented by the sequence of comparators added so far. This sequence is encoded into a fixed-size vector.
2.  **Actions:** The set of all possible comparators `(i, j)` that can be added to the network, where `i` and `j` are wire indices (`0 <= i < j < n`). The agent chooses one comparator at each step.
3.  **Transitions:** Adding the chosen comparator moves the state from the current sequence `S` to the new sequence `S + [comparator]`.
4.  **Rewards:** A reward signal guides the agent towards desirable outcomes (short, valid sorting networks). This implementation uses a **scaled terminal reward** strategy:
    *   **Success:** If adding a comparator results in a valid sorting network (checked using the Zero-One Principle), a large positive reward `R = success_base / num_steps` is given. Dividing by the number of steps (`num_steps`) explicitly encourages finding shorter solutions. The episode terminates.
    *   **Failure:** If the maximum allowed number of comparators (`max_steps`) is reached without forming a valid sorting network, a large negative reward (`failure`) is given. The episode terminates.
    *   **Intermediate Steps:** A small (potentially zero) penalty (`step_penalty`) can be given for each comparator added before termination.
5.  **Goal:** The agent aims to learn a policy (a strategy for choosing actions/comparators) that maximizes the expected cumulative discounted reward, effectively leading it to discover short, valid sorting networks.

### Approach: Deep Q-Learning

We employ the Deep Q-Network (DQN) algorithm, a popular value-based DRL method:

-   **Q-Value Function:** A neural network (`QNetwork`) is trained to approximate the optimal action-value function, Q*(s, a). Q*(s, a) represents the maximum expected future reward achievable by taking action `a` in state `s` and following the optimal policy thereafter.
-   **Policy:** The agent's policy is derived from the learned Q-values. Typically, it follows an **epsilon-greedy** strategy during training: with probability epsilon, it explores by choosing a random action; otherwise, it exploits by choosing the action `a` that maximizes Q(s, a) according to the current network. During evaluation, epsilon is set to 0 (pure exploitation).
-   **Experience Replay:** Transitions `(state, action, reward, next_state, done)` are stored in a replay buffer. The agent samples mini-batches from this buffer to train the Q-network, breaking correlations between consecutive experiences and improving sample efficiency.
-   **Target Network & Double DQN:** A separate `target_net` is used to generate stable target values for the Q-learning updates, mitigating oscillations. The Double DQN technique further reduces the overestimation bias common in standard DQN by decoupling action selection and value estimation in the target calculation.

## Features

-   **DRL-based Generation:** Learns to construct sorting networks without predefined algorithms.
-   **Size Optimization Focus:** Reward structure specifically incentivizes minimizing the number of comparators.
-   **PyTorch Implementation:** Built using the PyTorch deep learning framework.
-   **Configurability:** Easy adjustment of environment parameters, agent hyperparameters, network architecture, and reward settings via YAML files.
-   **Modularity:** Well-defined components (Environment, Agent, Model, Trainer, Evaluator, Utilities).
-   **Verification:** Includes robust checking of network validity using the Zero-One Principle.
-   **Pruning:** Utility function to remove redundant comparators from valid networks found by the agent.
-   **Visualization:** Text-based output showing the structure of generated networks.
-   **Checkpointing & Resuming:** Saves training state (model weights, epsilon, best network) and allows resuming interrupted training sessions.


## Project Structure

```
RLSortingNetworks/
├── configs/                 # Configuration files (YAML)
│   └── default_config.yaml
├── checkpoints/             # Default directory for saving run artifacts
│   └── <run_id>/            # Specific run directory (e.g., 4w_10s)
│       ├── config.yaml      # Copy of config used for this run
│       ├── model.pt         # Saved policy network weights
│       ├── epsilon.npy      # Saved epsilon value
│       ├── training.log     # Detailed training log file
│       ├── results.csv      # Episode summary results
│       └── best_network.csv # Best network found
├── sorting_network_rl/      # Main source code package (importable)
│   ├── __init__.py
│   ├── agent/               # RL Agent (DQN)
│   │   └── dqn_agent.py
│   ├── core/                # Core logic (training, evaluation loops)
│   │   ├── evaluator.py
│   │   └── trainer.py
│   ├── env/                 # Environment definition
│   │   └── sorting_env.py
│   ├── model/               # Neural network models (Q-Network)
│   │   └── network.py
│   └── utils/               # Helper functions and utilities
│       ├── config_loader.py # YAML config handling
│       ├── evaluation.py    # is_sorted, is_sorting_network, prune, visualize
│       ├── logging_setup.py # Logging configuration
│       └── state_encoder.py # State vector generation
├── scripts/                 # Executable scripts to run training/evaluation
│   ├── train.py
│   └── evaluate.py
└── requirements.txt         # Python package dependencies
```


## Configuration (`configs/default_config.yaml`)

This YAML file controls all major aspects of the training process.

```yaml
# Environment Settings
environment:
  n_wires: 4          # Number of inputs/outputs to sort
  max_steps: 10       # Max comparators allowed per episode (should be > optimal)

# Training Settings
training:
  num_episodes: 5000  # Total training episodes
  target_update_freq: 100 # How often to copy policy_net weights to target_net
  print_every: 50     # Log summary statistics every N episodes
  start_fresh: True   # If False, tries to load checkpoint from run_dir
  save_checkpoints: True # Save model/epsilon periodically

# DQN Agent Settings
agent:
  lr: 5.0e-4          # Learning rate for the Adam optimizer
  gamma: 0.95         # Discount factor for future rewards
  epsilon_start: 1.0  # Starting exploration rate
  epsilon_end: 0.01   # Minimum exploration rate
  epsilon_decay: 0.995 # Multiplicative decay factor per episode
  buffer_size: 10000  # Capacity of the experience replay buffer
  batch_size: 64      # Number of transitions sampled per training step

# Reward Structure (Scaled Terminal Reward)
reward:
  success_base: 100.0 # Base reward for finding a sorting network (scaled by 1/steps)
  failure: -10.0      # Penalty for reaching max_steps without sorting
  step_penalty: 0.0   # Penalty per step (e.g., -0.01 to discourage steps slightly)

# Model Architecture (Q-Network MLP)
model:
  fc1_units: 256      # Neurons in the first hidden layer
  fc2_units: 256      # Neurons in the second hidden layer

# Checkpoint/Logging Settings
experiment:
  base_dir: "checkpoints" # Parent directory for saving run data
  # run_id is generated automatically as {n_wires}w_{max_steps}s
```

## How It Works: Implementation Details

This section breaks down the core components of the `PySortNetRL` project and explains their roles and interactions.

### 1. The Environment (`sorting_network_rl.env.sorting_env`)

*   **Purpose:** Simulates the step-by-step construction of a sorting network. It provides the interface for the RL agent to interact with.
*   **Class:** `SortingNetworkEnv`
*   **Key Attributes:**
    *   `n_wires`: Number of input/output channels for the network.
    *   `max_steps`: Maximum number of comparators allowed in an episode.
    *   `_all_actions`: A list containing all possible unique comparator pairs `(i, j)` where `0 <= i < j < n_wires`. This defines the action space.
    *   `_action_to_comparator`: A dictionary mapping an integer action index to its corresponding `(i, j)` tuple.
    *   `comparators`: A list storing the sequence of comparators added in the current episode.
    *   `current_step`: The number of comparators added so far in the current episode.
*   **Core Methods:**
    *   `__init__(n_wires, max_steps)`: Initializes the environment, calculates the action space based on `n_wires`.
    *   `reset()`: Clears `self.comparators`, resets `self.current_step` to 0, and returns the initial observation (representing an empty network). Called at the beginning of each training episode.
    *   `step(action)`:
        1.  Takes an integer `action` index as input.
        2.  Looks up the corresponding comparator `(i, j)` using `_action_to_comparator`.
        3.  Appends the comparator to the `self.comparators` list.
        4.  Increments `self.current_step`.
        5.  Checks if the episode should end based on the step count (`done = self.current_step >= self.max_steps`).
        6.  Returns a tuple: `(observation, reward, done)`.
            *   `observation`: The new state (current comparators and step count).
            *   `reward`: Always returns `0.0` from the environment itself. The meaningful reward calculation happens in the `Trainer` based on the resulting state.
            *   `done`: Boolean indicating if the episode terminated due to reaching `max_steps`.
    *   `_get_observation()`: Returns a dictionary containing a *copy* of the current `comparators` list and the `current_step`.
    *   `get_action_space_size()`: Returns the total number of possible actions (comparators).

### 2. State Representation (`sorting_network_rl.utils.state_encoder`)

*   **Purpose:** To convert the variable-length list of comparators (the environment's natural state) into a fixed-size numerical vector that can be fed into the neural network.
*   **Function:** `encode_state(n_wires, max_steps, comparators)`
*   **Methodology:**
    1.  Calculates the total number of unique possible comparators (`num_possible_comparators = n_wires * (n_wires - 1) // 2`).
    2.  Creates a mapping (cached using `lru_cache`) from each comparator tuple `(i, j)` to a unique integer index from `0` to `num_possible_comparators - 1`.
    3.  Initializes a 2D NumPy array (matrix) of zeros with shape `(max_steps, num_possible_comparators)`.
    4.  Iterates through the input `comparators` list. For the comparator added at `step` (up to `max_steps - 1`):
        *   Finds its unique `index` from the mapping.
        *   Sets the element `matrix[step, index]` to `1.0`. This essentially marks "which" comparator was chosen at "which" step.
    5.  Flattens the 2D matrix into a 1D NumPy vector using `.flatten()`. The size of this vector is always `max_steps * num_possible_comparators`.
*   **Output:** A 1D `np.float32` array representing the state.

### 3. The Agent (`sorting_network_rl.agent.dqn_agent`)

*   **Purpose:** Implements the DQN learning algorithm and manages the agent's interaction policy.
*   **Class:** `DQNAgent`
*   **Key Components:**
    *   **Networks:** `policy_net` (instance of `QNetwork`, actively trained) and `target_net` (instance of `QNetwork`, weights periodically copied from `policy_net`). Both are moved to the appropriate device (CPU/GPU).
    *   **Optimizer:** `torch.optim.Adam` used to update the `policy_net` weights based on calculated gradients.
    *   **Replay Buffer:** `collections.deque(maxlen=buffer_size)` stores past experiences `(state, action, reward, next_state, done)`.
    *   **Hyperparameters:** `gamma` (discount factor), `epsilon` (current exploration rate), `epsilon_min`, `epsilon_decay`, `batch_size`.
*   **Core Methods:**
    *   `__init__(...)`: Initializes networks, optimizer, buffer, and hyperparameters based on the configuration dictionary. Calls `update_target_network()` initially to synchronize weights.
    *   `select_action(state_vector)`:
        *   Generates a random number. If less than `self.epsilon`, returns a random action index (exploration).
        *   Otherwise (exploitation):
            *   Converts the NumPy `state_vector` to a PyTorch tensor.
            *   Passes the tensor through `self.policy_net` (in evaluation mode) to get Q-values for all actions.
            *   Returns the index of the action with the highest Q-value (`.argmax().item()`).
    *   `store_transition(...)`: Appends the provided experience tuple to the `replay_buffer`. Ensures states are NumPy arrays.
    *   `_sample_batch()`: Randomly samples `self.batch_size` transitions from the `replay_buffer` and converts them into PyTorch tensors (states, actions, rewards, next_states, dones) on the correct device.
    *   `train_step()`:
        1.  Checks if the buffer has enough samples (`>= batch_size`). If not, returns `None`.
        2.  Calls `_sample_batch()` to get a batch of data.
        3.  **Calculate Current Q-Values:** Uses `policy_net` to compute `Q(s, a)` for the sampled states `s` and actions `a`. Uses `.gather(1, actions)` to select the specific Q-values for the actions actually taken.
        4.  **Calculate Target Q-Values (Double DQN):**
            *   Uses `policy_net` to find the *best action indices* `a'_max` for the `next_states` (`s'`). (`policy_net(next_states).argmax(1, keepdim=True)`).
            *   Uses `target_net` to evaluate the Q-value of *those specific actions* `a'_max` in the `next_states`. (`target_net(next_states).gather(1, policy_next_actions)`).
            *   Calculates the final target: `target = rewards + self.gamma * target_q_values * (1 - dones)`.
        5.  **Calculate Loss:** Computes the difference between `current_q` and `target_q` using Huber Loss (`nn.functional.smooth_l1_loss`).
        6.  **Optimize:** Performs backpropagation (`loss.backward()`) and updates `policy_net` weights (`optimizer.step()`). Optional gradient clipping is included.
        7.  Returns the scalar loss value.
    *   `update_target_network()`: Copies the state dictionary (weights and biases) from `policy_net` to `target_net`.
    *   `decay_epsilon()`: Multiplies `self.epsilon` by `self.epsilon_decay`, ensuring it doesn't fall below `self.epsilon_min`.
    *   `save_model()`, `load_model()`, `save_epsilon()`, `load_epsilon()`: Handle checkpointing operations.

### 4. The Q-Network Model (`sorting_network_rl.model.network`)

*   **Purpose:** To approximate the Q-value function Q(s, a).
*   **Class:** `QNetwork` (inherits `torch.nn.Module`)
*   **Architecture:** A feed-forward Multi-Layer Perceptron (MLP).
    *   **Input Layer:** Takes the flattened state vector from `encode_state`. Its size is `max_steps * num_possible_comparators`.
    *   **Hidden Layers:** One or more `torch.nn.Linear` layers followed by `torch.nn.functional.relu` activation functions. The number and size of these layers are defined in the `config['model']` section (e.g., `fc1_units`, `fc2_units`).
    *   **Output Layer:** A final `torch.nn.Linear` layer with `action_dim` output neurons. Each output neuron corresponds to a possible action (comparator), and its activation represents the estimated Q-value for taking that action in the input state.
*   **`forward(x)` method:** Defines how the input tensor `x` (batch of state vectors) flows through the layers to produce the output Q-values.

### 5. The Training Process (`sorting_network_rl.core.trainer`)

*   **Purpose:** Manages the high-level training loop, coordinating the agent and environment interactions, reward calculation, logging, and checkpointing.
*   **Class:** `Trainer`
*   **Key Responsibilities:**
    *   **Setup:** Initializes the environment, agent, and experiment directory structure (`_setup_paths`). Loads previous checkpoints if not starting fresh (`_load_checkpoint`). Saves the current run's configuration (`save_config`). Sets up file logging via `setup_logging`.
    *   **Main Loop (`train`)**: Iterates through the configured `num_episodes`.
    *   **Episode Loop:**
        *   Resets the environment `env.reset()`.
        *   Runs a loop until the episode ends (`done = True`).
        *   Inside the step loop: gets state, selects action (`agent.select_action`), takes step (`env.step`), gets next state.
        *   **Reward Calculation:** This is where the core reward logic resides. After `env.step`, the `Trainer` checks the *new* state:
            *   Calls `is_sorting_network()` on `next_state_info['comparators']`.
            *   If `True`: Assigns `reward = reward_cfg['success_base'] / num_steps`, sets `done = True`. Checks if it's a new best network and saves it (`_save_best_network`).
            *   If `False` and `env_done` (max steps reached): Assigns `reward = reward_cfg['failure']`, sets `done = True`.
            *   Otherwise: Assigns `reward = reward_cfg['step_penalty']`.
        *   Stores the transition `(state, action, reward, next_state, done)` using `agent.store_transition()`.
        *   Calls `agent.train_step()` to potentially update the agent's network.
        *   Updates the state for the next iteration.
    *   **Post-Episode:** Calls `agent.decay_epsilon()`, logs episode statistics (`_log_episode_results`), updates the target network (`agent.update_target_network()`) periodically, and saves checkpoints (`_save_checkpoint`) periodically.
    *   **Completion:** Logs final messages and potentially visualizes/prunes the best network found.

### 6. Evaluation & Validation (`sorting_network_rl.utils.evaluation`)

*   **Purpose:** To provide functions for checking the correctness and properties of generated networks.
*   **Key Functions:**
    *   `is_sorting_network(n_wires, comparators)`:
        *   Implements the **Zero-One Principle**.
        *   Iterates through all `2^n_wires` binary input sequences.
        *   For each input, simulates the network using `apply_comparators`.
        *   Checks if the output is sorted using `is_sorted`.
        *   Returns `False` immediately if any input fails; returns `True` if all inputs are sorted correctly.
    *   `apply_comparators(input_list, comparators)`: Simulates passing a single `input_list` through the `comparators`, performing swaps as needed. Returns the potentially modified list.
    *   `is_sorted(lst)`: Checks if a list `lst` is in non-decreasing order.

### 7. Pruning (`sorting_network_rl.utils.evaluation`)

*   **Purpose:** To simplify a *valid* sorting network by removing unnecessary comparators.
*   **Function:** `prune_redundant_comparators(n_wires, comparators)`
*   **Methodology:**
    1.  Takes a known *valid* sorting network as input.
    2.  Iterates through the comparators, typically **backwards** (from the last comparator to the first).
    3.  For each comparator:
        *   Temporarily remove it from a copy of the network list.
        *   Check if the *modified network* is *still* a valid sorting network using `is_sorting_network()`.
        *   If it is still valid, the comparator was redundant, so keep it removed.
        *   If it's no longer valid, the comparator was essential, so add it back to the list in its original position.
    4.  Returns the final list, which is guaranteed to be a valid sorting network and potentially shorter than the original.

### 8. Visualization (`sorting_network_rl.utils.evaluation`)

*   **Purpose:** To generate a simple text-based representation of the network structure.
*   **Function:** `format_network_visualization(comparators, n_wires)`
*   **Methodology:**
    *   Creates a grid where rows are wires and columns are steps/comparators.
    *   Draws horizontal lines (`─`) for each wire.
    *   For each comparator `(i, j)` at step `t`:
        *   Places connection points (`●`) at `(wire_i, step_t)` and `(wire_j, step_t)`.
        *   Draws vertical lines (`|`) between `wire_i` and `wire_j` at `step_t`.
    *   Formats this grid into a readable multi-line string.

### 9. Evaluation Script (`sorting_network_rl.core.evaluator`, `scripts/evaluate.py`)

*   **Purpose:** To load a trained agent or a saved network and analyze its performance/structure.
*   **`Evaluator` Class:**
    *   Loads the configuration and the trained `model.pt` for a specific run.
    *   Initializes an agent with the loaded weights and sets it to evaluation mode (epsilon=0).
    *   Provides `evaluate_policy()` to run the deterministic policy and `load_network_from_csv()` to load the best network saved during training.
*   **`scripts/evaluate.py` Script:**
    *   Sets hardcoded parameters (run directory, prune flag, eval_best flag).
    *   Validates paths.
    *   Loads the relevant config.
    *   Initializes the `Evaluator`.
    *   Calls either `evaluator.load_network_from_csv()` or `evaluator.evaluate_policy()` based on the `eval_best` flag.
    *   Calls `_analyze_and_visualize` helper function to print network steps, validity, visualization, and optionally prune the result.

### 10. Utilities (`sorting_network_rl.utils`)

*   **`config_loader`:** Reads (`load_config`) and writes (`save_config`) configuration dictionaries from/to YAML files. Ensures consistency and reproducibility.
*   **`logging_setup`:** Configures the Python `logging` framework to write timestamped messages with severity levels (INFO, DEBUG, ERROR) to both the console and a log file (`training.log`) specific to each run. Used primarily by the `Trainer`.


## Usage

This section explains how to run the training and evaluation scripts. Ensure you have activated the project's virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`) before running these commands.

### Training the Agent (`scripts/train.py`)

This script initiates the DQN agent training process based on the settings in a configuration file.

**1. Configure:**
   - Open `configs/default_config.yaml` (or create a copy).
   - Adjust parameters like `n_wires`, `max_steps`, learning rate (`lr`), epsilon decay (`epsilon_decay`), number of episodes (`num_episodes`), reward values, etc., according to your needs.
   - Pay special attention to `training.start_fresh`:
     - `True`: Ignores existing checkpoints in the target run directory and starts training from scratch. Log files and results might be overwritten.
     - `False`: Attempts to load `model.pt`, `epsilon.npy`, and determine the starting episode from `results.csv` within the corresponding run directory (`checkpoints/{n}w_{m}s/`). If found, training resumes; otherwise, it starts fresh.

**2. Run Training:**
   - Navigate to the project's root directory (`RLSortingNetworks`) in your terminal.
   - To use the default configuration:
     ```bash
     python scripts/train.py
     ```
   - To specify a different configuration file:
     ```bash
     python scripts/train.py -c path/to/your/custom_config.yaml
     ```

**3. Monitoring:**
   - The script will print summary statistics to the console every `training.print_every` episodes, including:
     - Current Episode number.
     - Steps taken in the last episode.
     - Total reward for the last episode.
     - Average reward and steps over the last `print_every` episodes.
     - Current Epsilon value.
     - Length of the best (shortest) valid sorting network found so far.
     - Last calculated training loss.
   - More detailed logs (including potential warnings or errors) are saved continuously to `checkpoints/<run_id>/training.log`.

**4. Output:**
   - A directory named `checkpoints/{n_wires}w_{max_steps}s/` will be created (or reused).
   - Inside this directory, you will find:
     - `config.yaml`: A copy of the configuration used for this run.
     - `model.pt`: Periodically saved weights of the trained policy network.
     - `epsilon.npy`: Periodically saved value of epsilon.
     - `results.csv`: A log of the summary statistics printed to the console.
     - `training.log`: Detailed logs from the Python `logging` module.
     - `best_network.csv`: If a valid sorting network is found, this file stores the shortest one encountered during training (updated whenever a shorter valid network is found).

### Evaluating a Trained Agent (`scripts/evaluate.py`)

This script loads a trained model or a saved network and performs analysis, visualization, and optional pruning. It's designed primarily for execution from an IDE by modifying hardcoded parameters.

**1. Configure `scripts/evaluate.py`:**
   - Open the `scripts/evaluate.py` file in your editor.
   - Locate the section `--- Hardcoded Configuration for IDE Execution ---`.
   - Modify the following variables:
     - `HARDCODED_RUN_DIR`: Set this string to the path (relative to the project root) of the specific run directory you want to evaluate. Example: `"checkpoints/4w_10s"`.
     - `HARDCODED_PRUNE`: Set to `True` if you want the script to attempt pruning the evaluated network, or `False` otherwise. Pruning only applies if the network is valid.
     - `HARDCODED_EVAL_BEST`:
       - Set to `True` to load, analyze, and visualize the network stored in the run directory's `best_network.csv` file.
       - Set to `False` to load the trained `model.pt` and run the agent's deterministic policy (epsilon=0) to generate, analyze, and visualize the resulting network.

**2. Run Evaluation:**
   - Execute the script directly from your IDE (e.g., Right-click -> Run 'evaluate.py') or from the terminal (while in the project root and with the virtual environment activated):
     ```bash
     python scripts/evaluate.py
     ```

**3. Output:**
   - The script will print information to the console:
     - Confirmation of the evaluation mode (hardcoded settings).
     - The source of the network being analyzed (Agent Policy or Best Network CSV).
     - A step-by-step list of the comparators in the network.
     - The length of the network.
     - Whether the network is `VALID` or `INVALID` (based on `is_sorting_network`).
     - A text-based visualization of the original network.
     - If `HARDCODED_PRUNE` was `True` and the network was valid:
       - A message indicating whether pruning found redundant comparators.
       - If successful, the reduced length and a visualization of the pruned network.

## Results and Checkpoints

-   All artifacts generated during a training run are stored within a specific subdirectory inside the `experiment.base_dir` (default: `checkpoints/`).
-   The subdirectory name is determined solely by the environment parameters: `{n_wires}w_{max_steps}s` (e.g., `4w_10s`). This allows for easy identification and resumption of runs with the same core parameters.
-   **Key Files within `<run_id>/`:**
    *   `config.yaml`: The exact configuration used for the run, ensuring reproducibility.
    *   `model.pt`: The saved state dictionary of the `policy_net`. This represents the agent's learned policy at the time of the last checkpoint.
    *   `epsilon.npy`: The value of the exploration parameter `epsilon` at the time of the last checkpoint. Used for resuming training.
    *   `best_network.csv`: Contains the sequence of comparators forming the *shortest valid* sorting network discovered *during the entire training process* up to the last update. This can sometimes be better than the network generated by the final `model.pt` policy.
    *   `results.csv`: A tabular log of episode summaries (reward, steps, epsilon, best length, loss) useful for plotting learning curves or analyzing trends.
    *   `training.log`: Verbose logs generated by the Python `logging` module, useful for debugging and detailed progress tracking.

### Performance Summary (Experimental Results)

This table summarizes the length of the shortest valid sorting networks found by this DRL agent during training runs with the default configuration (or specify which config). Results may vary depending on hyperparameters and training duration.

| `n` | `max_steps` (Used) | Optimal Size | Optimal Depth | Agent Best Size | Agent Pruned Size | Agent Depth | Notes / Config ID |
| :-: | :----------------: |:------------:| :-----------: |:---------------:|:-----------------:|:-----------:|:------------------|
| 1   | 1                  |      0       | 0             |       *0*       |        *0*        |     *0*     | (Trivial)         |
| 2   | 2                  |      1       | 1             |       *1*       |                   |     *1*     | `2w_2s`           |
| 3   | 5                  |      3       | 3             |       *3*       |                   |     *3*     | `3w_5s`           |
| 4   | 10                 |      5       | 3             |       *6*       |        *5*        |     *8*     | `4w_10s`          |
| 5   | 15                 |      9       | 5             |      *10*       |        *9*        |     *7*     | `5w_15s`          |
| 6   | 20                 |      12      | 5             |      *18*       |       *13*        |     *9*     | `6w_25s`          |
| 7   | 25                 |      16      | 6             |      *33*       |       *17*        |    *10*     | `7w_25s`          |
| 8   | 30                 |      19      | 6             |       *?*       |        *?*        |     *?*     | `8w_30s`          |
| 9   | 35                 |      25      | 7             |       *?*       |        *?*        |     *?*     | `9w_35s`          |
| 10  | 40                 |      29      | 7             |       *?*       |        *?*        |     *?*     | `10w_40s`         |
| 11  | 45                 |      35      | 8             |       *?*       |        *?*        |     *?*     | `11w_45s`         |
| 12  | 50                 |      39      | 8             |       *?*       |        *?*        |     *?*     | `12w_50s`         |
| 13  | 55                 |   43 - 45    | 9             |       *?*       |        *?*        |     *?*     | `13w_55s`         |
| 14  | 60                 |   47 - 51    | 9             |       *?*       |        *?*        |     *?*     | `14w_60s`         |
| 15  | 65                 |   51 - 56    | 9             |       *?*       |        *?*        |     *?*     | `15w_65s`         |
| 16  | 70                 |   55 - 60    | 9             |       *?*       |        *?*        |     *?*     | `16w_70s`         |
| 17  | 80                 |   60 - 71    | 10            |       *?*       |        *?*        |     *?*     | `17w_80s`         |

**Notes:**
*   "Best Network Length (Agent Found)" refers to the shortest valid network saved in `best_network.csv` during training.
*   "Pruned Network Length" is the result after applying the `prune_redundant_comparators` function to the best network found.
*   "Optimal Known Size" is listed for reference (see [State of the Art](#state-of-the-art-sorting-networks) if included, or external sources).
*   Results were obtained using run configurations identified by the Config ID (e.g., `3w_5s` corresponds to the directory `checkpoints/3w_5s/`). Ensure the config matches if reproducing.

## Future Work

This project provides a foundation for using DRL to find sorting networks. Potential areas for future development include:

-   **Depth Optimization:** Modify the reward structure or state representation to explicitly encourage networks with lower depth (fewer parallel steps), possibly alongside size minimization.
-   **Improved State Representation:** Explore more sophisticated state representations that might better capture the network's structure and sorting progress, such as Graph Neural Networks (GNNs).
-   **Action Masking:** Prevent the agent from selecting comparators that are known to be redundant or sub-optimal in certain states (e.g., comparing already sorted adjacent wires).
-   **Hyperparameter Tuning:** Employ automated hyperparameter optimization techniques (e.g., using Optuna, Ray Tune) to find optimal settings for learning rate, network architecture, epsilon decay, etc.
-   **Curriculum Learning:** Start training with smaller `n_wires` and gradually increase the problem difficulty, potentially transferring knowledge learned on simpler tasks.
-   **Parallel Environments:** Utilize libraries like `multiprocessing` or Ray to run multiple environment instances in parallel, significantly speeding up experience collection and training time.
