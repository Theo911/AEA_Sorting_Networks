import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import logging

class QNetwork(nn.Module):
    """Deep Q-Network (DQN) model for predicting Q-values for actions."""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initializes the Q-network.

        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Number of possible actions (comparators).
            config (Dict[str, Any]): Configuration dictionary containing model parameters
                                      (e.g., 'model.fc1_units', 'model.fc2_units').
        """
        super().__init__()
        fc1_units = config.get('model', {}).get('fc1_units', 256)
        fc2_units = config.get('model', {}).get('fc2_units', 256)

        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)
        logging.info(f"Initialized QNetwork with layers: Linear({state_dim}, {fc1_units}), ReLU, Linear({fc1_units}, {fc2_units}), ReLU, Linear({fc2_units}, {action_dim})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Q-values for each possible action, shape (batch_size, action_dim).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values