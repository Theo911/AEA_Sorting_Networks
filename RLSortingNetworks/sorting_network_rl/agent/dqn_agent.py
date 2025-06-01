import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Deque, Dict, Any, Optional, List
import os

from RLSortingNetworks.sorting_network_rl.model.network import QNetwork

logger = logging.getLogger(__name__)

class DQNAgent:
    """Deep Q-Learning Agent implementing Double DQN.

    Attributes:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        device (torch.device): Device (CPU or CUDA) for tensor computations.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Current exploration rate (epsilon-greedy).
        epsilon_min (float): Minimum value for epsilon.
        epsilon_decay (float): Multiplicative decay factor for epsilon.
        batch_size (int): Size of batches sampled from the replay buffer.
        policy_net (QNetwork): The main Q-network being trained.
        target_net (QNetwork): The target Q-network (periodically updated).
        optimizer (optim.Optimizer): Optimizer for the policy network.
        replay_buffer (Deque): Experience replay buffer storing transitions.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initializes the DQN agent.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Number of possible actions.
            config (Dict[str, Any]): Configuration dictionary containing agent parameters
                                      (lr, gamma, epsilon_*, buffer_size, batch_size)
                                      and model parameters.
        """
        agent_cfg = config.get('agent', {})
        self.state_dim = state_dim
        self.action_dim = action_dim

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("MPS (Apple Silicon GPU) is available. Using device: mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDA is available. Using device: cuda")
        else:
            self.device = torch.device("cpu")
            logger.info("Neither MPS nor CUDA is available. Using device: cpu")
        logger.info(f"Using device: {self.device}")

        self.gamma = agent_cfg.get('gamma', 0.99)
        self.epsilon = agent_cfg.get('epsilon_start', 1.0)
        self.epsilon_min = agent_cfg.get('epsilon_end', 0.01)
        self.epsilon_decay = agent_cfg.get('epsilon_decay', 0.995)
        self.batch_size = agent_cfg.get('batch_size', 64)
        buffer_size = agent_cfg.get('buffer_size', 10000)

        self.replay_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=buffer_size)

        # Initialize policy and target networks
        self.policy_net = QNetwork(state_dim, action_dim, config).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, config).to(self.device)
        self.update_target_network() # Initialize target_net weights same as policy_net
        self.target_net.eval() # Target network is only for inference

        learning_rate = agent_cfg.get('lr', 1e-3)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        logger.info(f"Initialized DQNAgent: state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"Hyperparameters: gamma={self.gamma}, epsilon_start={self.epsilon}, "
                    f"epsilon_end={self.epsilon_min}, epsilon_decay={self.epsilon_decay}, "
                    f"batch_size={self.batch_size}, buffer_size={buffer_size}, lr={learning_rate}")

    def select_action(self,
                      state_vector: np.ndarray,
                      invalid_action_indices: Optional[List[int]] = None
                      ) -> int:
        """Selects an action using an epsilon-greedy policy, applying action masking.

        Args:
            state_vector (np.ndarray): The current state represented as a NumPy array.
            invalid_action_indices (Optional[List[int]]): A list of action indices
                that should be masked (not chosen during exploitation). Defaults to None.

        Returns:
            int: The index of the selected (potentially masked) action.
        """
        # --- Exploration Phase ---
        if random.random() < self.epsilon:
            # Get all possible action indices
            possible_actions = list(range(self.action_dim))
            if invalid_action_indices:
                # Filter out the invalid actions to get the set of valid actions
                valid_actions = [a for a in possible_actions if a not in invalid_action_indices]
                # If, somehow, all actions are masked as invalid (unlikely),
                # choose randomly from the original set to avoid an error and log a warning.
                if not valid_actions:
                    logger.warning("Exploration: All actions were masked as invalid? Choosing from original set.")
                    return random.choice(possible_actions)
                # Choose a random action from the *valid* subset
                return random.choice(valid_actions)
            else:
                # If no invalid actions are specified, choose randomly from all possible actions
                return random.choice(possible_actions)

        # --- Exploitation Phase ---
        else:
            # Convert state numpy array to a PyTorch tensor, add batch dimension, move to device
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(self.device)
            # Set the policy network to evaluation mode (disables dropout, etc.) for inference
            self.policy_net.eval()
            # Disable gradient calculations for inference
            with torch.no_grad():
                # Get Q-values for all actions in the current state.
                # Squeeze [0] to remove the batch dimension, result shape: [action_dim]
                q_values = self.policy_net(state_tensor)[0]

            # --- Apply Action Masking ---
            if invalid_action_indices:
                # Convert list of invalid indices to a tensor for efficient indexing
                invalid_indices_tensor = torch.LongTensor(invalid_action_indices).to(self.device)

                # Ensure indices are within the valid range before masking
                # This prevents potential index out of bounds errors
                valid_mask_indices = invalid_indices_tensor[invalid_indices_tensor < self.action_dim]

                # If there are any valid indices to mask
                if len(valid_mask_indices) > 0:
                    # Set the Q-values for the invalid actions to negative infinity.
                    # This ensures they won't be selected by argmax.
                    q_values[valid_mask_indices] = -torch.inf
                    # logger.debug(f"Masked actions: {valid_mask_indices.tolist()}") # Optional debug log

                # Check if all actions were masked (all Q-values are -infinity)
                # This is an edge case, could happen if masking logic is too aggressive
                if torch.all(q_values == -torch.inf):
                    logger.warning(
                        "Exploitation: All actions were masked! Choosing a random valid action (if any) to proceed.")
                    # Fallback: Choose a random valid action if possible, otherwise any action.
                    possible_actions = list(range(self.action_dim))
                    # Re-filter valid actions based on the original invalid list
                    valid_actions = [a for a in possible_actions if a not in invalid_action_indices]
                    if valid_actions:
                        best_action = random.choice(valid_actions)
                    else:  # Extreme fallback if even filtering fails
                        logger.warning(
                            "Exploitation: No valid actions found after masking all. Choosing from original set.")
                        best_action = random.choice(possible_actions)
                else:
                    # Normal case: Find the action index with the maximum Q-value among the *valid* actions
                    best_action = q_values.argmax().item()
            else:
                # If no invalid actions were provided, choose the action with the highest Q-value overall
                best_action = q_values.argmax().item()
            # --- End Action Masking ---

            # Set the policy network back to training mode
            self.policy_net.train()
            # Return the selected action index
            return best_action

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Stores an experience transition in the replay buffer.

        Args:
            state (np.ndarray): The state before the action.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The state after the action.
            done (bool): Whether the episode terminated after this transition.
        """
        # Ensure inputs are NumPy arrays if they weren't already
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def _sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a batch of transitions from the replay buffer and converts them to tensors.

        Returns:
            Tuple containing tensors for: states, actions, rewards, next_states, dones.
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of numpy arrays to single numpy arrays before making tensors
        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64) # Actions are indices
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32) # Use float for multiplication (1-dones)

        # Convert numpy arrays to tensors and move to the appropriate device
        states_tensor = torch.from_numpy(states_np).to(self.device)
        actions_tensor = torch.from_numpy(actions_np).unsqueeze(1).to(self.device) # Shape: [batch_size, 1]
        rewards_tensor = torch.from_numpy(rewards_np).unsqueeze(1).to(self.device) # Shape: [batch_size, 1]
        next_states_tensor = torch.from_numpy(next_states_np).to(self.device)
        dones_tensor = torch.from_numpy(dones_np).unsqueeze(1).to(self.device) # Shape: [batch_size, 1]

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def train_step(self) -> Optional[float]:
        """Performs one training step using a batch sampled from the replay buffer.

        Implements the Double DQN update rule:
        Q_target = r + gamma * Q_target_net(s', argmax_a' Q_policy_net(s', a'))

        Returns:
            Optional[float]: The calculated loss value for this step, or None if
                             the buffer doesn't have enough samples yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough samples to train yet

        states, actions, rewards, next_states, dones = self._sample_batch()

        # --- Calculate current Q-values ---
        # Get Q(s, a) for the actions taken using the policy network
        current_q = self.policy_net(states).gather(1, actions)

        # --- Calculate target Q-values using Double DQN ---
        with torch.no_grad():
            # 1. Select the best action a' for next_states s' using the policy network
            policy_next_actions = self.policy_net(next_states).argmax(1, keepdim=True) # Shape: [batch_size, 1]

            # 2. Evaluate the Q-value of that action a' using the target network: Q_target(s', a')
            max_next_q = self.target_net(next_states).gather(1, policy_next_actions)

            # Calculate the target Q-value: R + gamma * Q_target(s', a') * (1 - done)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # --- Calculate loss ---
        # Use Mean Squared Error (MSE) loss, or Huber loss for more robustness
        # loss = nn.functional.mse_loss(current_q, target_q)
        loss = nn.functional.smooth_l1_loss(current_q, target_q) # Huber loss

        # --- Optimize the policy network ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient Clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Copies the weights from the policy network to the target network."""
        logger.debug("Updating target network weights.")
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decays the exploration rate (epsilon) multiplicatively."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, file_path: str) -> None:
        """Saves the policy network's state dictionary."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(self.policy_net.state_dict(), file_path)
            logger.info(f"Policy network saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {e}")

    # def load_model(self, file_path: str) -> None:
    #     """Loads the policy network's state dictionary."""
    #     if not os.path.exists(file_path):
    #         logger.error(f"Model file not found: {file_path}")
    #         raise FileNotFoundError(f"Model file not found: {file_path}")
    #     try:
    #         # Load state dict, ensuring correct device mapping
    #         self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
    #         self.policy_net.to(self.device) # Ensure model is on the correct device
    #         self.update_target_network() # Also update target network upon loading
    #         self.policy_net.train() # Set to train mode by default after loading
    #         self.target_net.eval()
    #         logger.info(f"Policy network loaded from {file_path}")
    #     except Exception as e:
    #         logger.error(f"Error loading model from {file_path}: {e}")
    #         raise IOError(f"Error loading model from {file_path}: {e}")

    def load_model(self, file_path: str) -> None:
        """Loads the policy network's state dictionary,
           attempting to adapt from older model formats if necessary.
        """
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")

        try:
            # Load the saved state dictionary from the file
            loaded_state_dict = torch.load(file_path, map_location=self.device)
            # Get the state dictionary of the current (potentially new) model architecture
            current_model_keys = set(self.policy_net.state_dict().keys())
            loaded_keys = set(loaded_state_dict.keys())

            # --- Adaptation Logic for Old Model Structure ---
            # This logic specifically handles the case where the old model's output layer
            # was named 'fc3' and the new model (with 2 hidden layers) names it 'fc_out'.
            # It assumes fc1 and fc2 layers are consistent.

            # Condition 1: Current model expects 'fc_out' but not a hidden 'fc3'.
            # This is true if 'fc_out.weight' is a key in current model, and 'fc3.weight' is NOT
            # (or if self.policy_net.fc3 attribute is None, based on your QNetwork init logic).
            current_expects_fc_out_from_fc2 = ("fc_out.weight" in current_model_keys and
                                               (not hasattr(self.policy_net, 'fc3') or self.policy_net.fc3 is None))

            # Condition 2: Loaded model has 'fc3' as output and no 'fc_out'.
            loaded_has_fc3_as_output = ("fc3.weight" in loaded_keys and
                                        "fc_out.weight" not in loaded_keys)

            if current_expects_fc_out_from_fc2 and loaded_has_fc3_as_output:
                logger.info(
                    f"Attempting to adapt legacy model format (fc3 as output layer) from {file_path} "
                    f"to current format (fc_out as output layer for 2-hidden-layer models)."
                )
                # Create a new state dict for the current model structure
                adapted_state_dict = {}
                keys_remapped = False
                for key, value in loaded_state_dict.items():
                    if key == "fc3.weight":
                        adapted_state_dict["fc_out.weight"] = value
                        keys_remapped = True
                    elif key == "fc3.bias":
                        adapted_state_dict["fc_out.bias"] = value
                        keys_remapped = True
                    elif key in current_model_keys:  # Copy matching keys (fc1, fc2)
                        adapted_state_dict[key] = value
                    else:
                        logger.warning(f"Key '{key}' from saved model not found in current model structure. Skipping.")

                if not keys_remapped:
                    logger.warning(
                        "Adaptation logic triggered, but no keys were actually remapped. Check model structures.")
                    # Fallback to loading original if no remapping occurred but conditions were met
                    final_state_dict_to_load = loaded_state_dict
                else:
                    # Check for missing keys in the adapted dict
                    missing_in_adapted = [k for k in current_model_keys if k not in adapted_state_dict]
                    if missing_in_adapted:
                        logger.error(f"Adaptation failed. Missing keys in adapted state_dict: {missing_in_adapted}")
                        raise RuntimeError(
                            f"State dict adaptation failed for {file_path}. Missing keys: {missing_in_adapted}")
                    final_state_dict_to_load = adapted_state_dict

            else:
                # No specific adaptation rule matched, load as is
                logger.debug(f"No specific adaptation rule matched for {file_path}. Attempting direct load.")
                final_state_dict_to_load = loaded_state_dict
            # --- END: Adaptation Logic ---

            # Load the (potentially adapted) state dictionary into the policy network
            self.policy_net.load_state_dict(final_state_dict_to_load)
            self.policy_net.to(self.device)  # Ensure model is on the correct device

            # Update the target network with the loaded policy network weights
            if hasattr(self, 'update_target_network'):  # Check if method exists (for DQNAgent)
                self.update_target_network()
            if hasattr(self, 'target_net') and self.target_net is not None:  # Check if attribute exists
                self.target_net.eval()  # Set target network to evaluation mode

            self.policy_net.train()  # Set policy network to train mode by default after loading
            logger.info(f"Policy network successfully loaded from {file_path}")

        except RuntimeError as e:
            # Catch specific PyTorch errors related to state_dict loading
            logger.error(f"RuntimeError loading state_dict for QNetwork from {file_path}: {e}")
            logger.error(
                "This is often due to a mismatch between the saved model's architecture "
                "and the current QNetwork class definition. Ensure the model structure "
                "(number/names of layers, sizes) matches or adapt the 'load_model' method."
            )
            # Re-raise as a more general IOError or a custom exception
            raise IOError(
                f"Failed to load model due to architecture mismatch or corrupted file: {file_path}. Original error: {e}")
        except Exception as e:
            # Catch any other unexpected errors during loading
            logger.error(f"An unexpected error occurred while loading model from {file_path}: {e}", exc_info=True)
            raise IOError(f"Failed to load model from {file_path}: {e}")

    def save_epsilon(self, file_path: str) -> None:
        """Saves the current epsilon value."""
        try:
            np.save(file_path, np.array(self.epsilon))
        except Exception as e:
            logger.error(f"Error saving epsilon to {file_path}: {e}")

    def load_epsilon(self, file_path: str) -> None:
        """Loads the epsilon value."""
        if not os.path.exists(file_path):
            logger.warning(f"Epsilon file not found: {file_path}. Using default.")
            return
        try:
            self.epsilon = float(np.load(file_path))
            logger.info(f"Epsilon loaded from {file_path}: {self.epsilon}")
        except Exception as e:
            logger.error(f"Error loading epsilon from {file_path}: {e}")