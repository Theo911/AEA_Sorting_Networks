import logging
from itertools import combinations
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SortingNetworkEnv:
    """Reinforcement Learning environment for building sorting networks.

    The agent interacts with this environment by adding comparators one by one.
    The goal is to find a sequence of comparators that sorts all possible inputs.
    """

    def __init__(self, n_wires: int, max_steps: int):
        """Initializes the sorting network environment.

        Args:
            n_wires (int): The number of wires (channels) the network should sort.
                           Must be >= 2.
            max_steps (int): The maximum number of comparators allowed in a network
                             (defines the maximum episode length).
        """
        if n_wires < 2:
            raise ValueError("Number of wires (n_wires) must be at least 2.")
        if max_steps <= 0:
            raise ValueError("Maximum steps (max_steps) must be positive.")

        self.n_wires = n_wires
        self.max_steps = max_steps

        # Generate all possible unique comparators (actions)
        self._all_actions: List[Tuple[int, int]] = list(combinations(range(n_wires), 2))
        self._action_to_comparator: Dict[int, Tuple[int, int]] = {i: comp for i, comp in enumerate(self._all_actions)}
        self._num_actions = len(self._all_actions)

        logger.info(f"Initialized SortingNetworkEnv: n_wires={n_wires}, max_steps={max_steps}, num_actions={self._num_actions}")

        # State variables reset in reset()
        self.comparators: List[Tuple[int, int]] = []
        self.current_step: int = 0

    def reset(self) -> Dict[str, Any]:
        """Resets the environment to its initial state (empty network).

        Returns:
            Dict[str, Any]: The initial state observation containing the empty
                           list of comparators and the current step (0).
        """
        self.comparators = []
        self.current_step = 0
        # logger.debug("Environment reset.")
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool]:
        """Executes one step in the environment by adding a comparator.

        Args:
            action (int): The index of the action (comparator) to add. Must be
                          within the valid range [0, num_actions - 1].

        Returns:
            Tuple[Dict[str, Any], float, bool]: A tuple containing:
                - observation (Dict[str, Any]): The new state observation.
                - reward (float): The reward obtained (typically 0 in this setup,
                                  calculated externally based on state).
                - done (bool): True if the episode has ended (max_steps reached),
                               False otherwise.

        Raises:
            ValueError: If the provided action index is invalid.
        """
        if not (0 <= action < self._num_actions):
            raise ValueError(f"Invalid action index {action}. Must be between 0 and {self._num_actions - 1}.")

        comparator_to_add = self._action_to_comparator[action]
        self.comparators.append(comparator_to_add)
        self.current_step += 1

        # Determine if the episode is done
        done = self.current_step >= self.max_steps

        # Reward is calculated based on the state *after* the step in the training loop.
        # We return a placeholder reward (e.g., 0) from the step function itself.
        reward = 0.0

        # logger.debug(f"Step {self.current_step}: Action {action} -> Add {comparator_to_add}, Done: {done}")

        return self._get_observation(), reward, done

    def _get_observation(self) -> Dict[str, Any]:
        """Constructs the current state observation dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing the current list of
                           'comparators' and the 'current_step' number.
        """
        return {
            'comparators': self.comparators.copy(), # Return a copy
            'current_step': self.current_step
        }

    def get_action_space_size(self) -> int:
        """Returns the total number of possible actions (comparators)."""
        return self._num_actions

    def get_comparators(self) -> List[Tuple[int, int]]:
        """Returns a copy of the current list of comparators."""
        return self.comparators.copy()

    def get_current_step(self) -> int:
        """Returns the current step number in the episode."""
        return self.current_step