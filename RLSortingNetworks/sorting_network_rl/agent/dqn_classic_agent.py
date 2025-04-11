import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Deque, Dict, Any, Optional
import os

from RLSortingNetworks.sorting_network_rl.model.network import QNetwork

logger = logging.getLogger(__name__)

class DQNAgent_Classic:
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
        optimizer (optim.Optimizer): Optimizer for the policy network.
        replay_buffer (Deque): Experience replay buffer storing transitions.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initializes the Classic DQN agent.

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # self.target_net = QNetwork(state_dim, action_dim, config).to(self.device)
        # self.update_target_network() # Initialize target_net weights same as policy_net
        # self.target_net.eval() # Target network is only for inference

        learning_rate = agent_cfg.get('lr', 1e-3)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        logger.info(f"Initialized DQNAgent: state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"Hyperparameters: gamma={self.gamma}, epsilon_start={self.epsilon}, "
                    f"epsilon_end={self.epsilon_min}, epsilon_decay={self.epsilon_decay}, "
                    f"batch_size={self.batch_size}, buffer_size={buffer_size}, lr={learning_rate}")

    def select_action(self, state_vector: np.ndarray) -> int:
        """Selects an action using an epsilon-greedy policy.

        Args:
            state_vector (np.ndarray): The current state represented as a NumPy array.

        Returns:
            int: The index of the selected action.
        """
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_dim)
        else:
            # Exploitation: choose the best action according to the policy network
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(self.device)
            self.policy_net.eval() # Set network to evaluation mode for inference
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            self.policy_net.train() # Set network back to train mode
            return q_values.argmax().item() # Get action with highest Q-value

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
        """Performs one training step using CLASSIC DQN update rule.

        Q_target = r + gamma * max_a' Q_policy_net(s', a')
        """
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough samples to train yet

        states, actions, rewards, next_states, dones = self._sample_batch()

        # --- Calculate current Q-values ---
        # Get Q(s, a) for the actions taken using the policy network
        current_q = self.policy_net(states).gather(1, actions)

        # --- Calculate target Q-values using Classic DQN ---
        with torch.no_grad():
            # Find the maximum Q-value for the next states using the SAME policy_net
            # max(1) returns values and indices; [0] gets the max values
            max_next_q = self.policy_net(next_states).max(1, keepdim=True)[0]

            # Calculate the target Q-value: R + gamma * max_a' Q_policy(s', a') * (1 - done)
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

    def load_model(self, file_path: str) -> None:
        """Loads the policy network's state dictionary."""
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")
        try:
            # Load state dict, ensuring correct device mapping
            self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
            self.policy_net.to(self.device) # Ensure model is on the correct device
            self.policy_net.train() # Set to train mode by default after loading
            logger.info(f"Policy network loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}")
            raise IOError(f"Error loading model from {file_path}: {e}")

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