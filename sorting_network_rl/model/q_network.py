import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for learning optimal sorting networks.
    """
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initializes the Q-network.

        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Number of possible actions (comparators).
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state tensor of shape (batch_size, state_dim)

        Returns:
            Tensor: Q-values for each possible action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
