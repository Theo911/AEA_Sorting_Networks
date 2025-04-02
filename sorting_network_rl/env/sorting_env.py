import numpy as np
from itertools import combinations
from sorting_network_rl.utils.evaluation import is_sorting_network

class SortingNetworkEnv:
    """
    Environment for learning sorting networks using reinforcement learning.
    """

    def __init__(self, n_wires: int, max_steps: int = 30):
        """
        Parameters:
            n_wires (int): Number of wires (channels)
            max_steps (int): Max number of comparators allowed
        """
        self.n = n_wires
        self.max_steps = max_steps
        self.all_actions = list(combinations(range(n_wires), 2))
        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns:
            state (dict): Initial environment state
        """
        self.comparators = []
        self.steps = 0
        return self.get_state()

    def step(self, action):
        """
        Apply a comparator and update environment.
        Parameters:
            action (int): Index of comparator in action space
        Returns:
            state (dict), reward (float), done (bool)
        """
        comparator = self.all_actions[action]
        self.comparators.append(comparator)
        self.steps += 1

        done = False
        reward = -1  # negative reward per step

        if is_sorting_network(self.n, self.comparators):
            reward = 100
            done = True
        elif self.steps >= self.max_steps:
            reward = -50
            done = True

        return self.get_state(), reward, done

    def get_state(self):
        """
        Returns the current state encoding.
        """
        return {
            'comparators': self.comparators.copy(),
            'steps': self.steps
        }

    def get_action_space_size(self):
        return len(self.all_actions)