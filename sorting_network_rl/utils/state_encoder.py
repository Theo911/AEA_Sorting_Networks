from typing import List, Tuple
import numpy as np

def encode_state(n_wires: int, max_steps: int, comparators: List[Tuple[int, int]]) -> np.ndarray:
    """
    Encodes the current state of the sorting network as a flat vector.

    Args:
        n_wires (int): Number of wires (channels).
        max_steps (int): Maximum allowed number of comparators.
        comparators (List[Tuple[int, int]]): List of comparator pairs (i, j).

    Returns:
        np.ndarray: Encoded state as a 1D float array of size (max_steps * num_possible_comparators).
    """
    num_possible = n_wires * (n_wires - 1) // 2
    state_matrix = np.zeros((max_steps, num_possible), dtype=np.float32)

    comparator_index_map = {}
    idx = 0
    for i in range(n_wires):
        for j in range(i + 1, n_wires):
            comparator_index_map[(i, j)] = idx
            idx += 1

    for step, (i, j) in enumerate(comparators):
        if step >= max_steps:
            break
        index = comparator_index_map[(i, j)]
        state_matrix[step, index] = 1.0

    return state_matrix.flatten()