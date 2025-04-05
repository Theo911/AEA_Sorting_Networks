from typing import List, Tuple, Dict
import numpy as np
import functools

@functools.lru_cache(maxsize=128) # Cache mapping for slight performance gain
def _get_comparator_index_map(n_wires: int) -> Dict[Tuple[int, int], int]:
    """Creates a mapping from (wire_i, wire_j) pairs to a unique index."""
    comparator_index_map: Dict[Tuple[int, int], int] = {}
    idx = 0
    for i in range(n_wires):
        for j in range(i + 1, n_wires):
            comparator_index_map[(i, j)] = idx
            idx += 1
    return comparator_index_map

def encode_state(n_wires: int, max_steps: int, comparators: List[Tuple[int, int]]) -> np.ndarray:
    """Encodes the current state of the sorting network as a flat vector.

    The state is represented as a matrix where each row corresponds to a step
    (up to max_steps) and each column corresponds to a possible comparator pair.
    A '1' indicates that the comparator was chosen at that step. The matrix
    is then flattened.

    Args:
        n_wires (int): Number of wires (channels).
        max_steps (int): Maximum allowed number of comparators (determines rows).
        comparators (List[Tuple[int, int]]): List of comparator pairs (i, j) chosen so far.

    Returns:
        np.ndarray: Encoded state as a 1D float32 NumPy array.
                    Shape: (max_steps * num_possible_comparators,).
    """
    num_possible_comparators = n_wires * (n_wires - 1) // 2
    state_matrix = np.zeros((max_steps, num_possible_comparators), dtype=np.float32)

    comparator_index_map = _get_comparator_index_map(n_wires)

    for step, comp in enumerate(comparators):
        if step >= max_steps:
            break
        # Ensure comparator is in canonical order (smaller_index, larger_index)
        canonical_comp = tuple(sorted(comp))
        if canonical_comp in comparator_index_map:
            index = comparator_index_map[canonical_comp]
            state_matrix[step, index] = 1.0
        # else: # Should not happen if comparators are generated correctly
        #     logging.warning(f"Comparator {comp} not found in index map for n_wires={n_wires}")

    return state_matrix.flatten()