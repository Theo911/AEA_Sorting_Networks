import itertools
import logging
from typing import List, Tuple, Sequence
import numpy as np # Added for type hint consistency if needed elsewhere

# Configure logging for this module
logger = logging.getLogger(__name__)

def apply_comparators(input_sequence: Sequence[int], comparators: List[Tuple[int, int]]) -> List[int]:
    """Applies a sequence of comparators to an input sequence (list or tuple).

    Args:
        input_sequence (Sequence[int]): Binary (0/1) input vector (list or tuple).
        comparators (List[Tuple[int, int]]): List of (i, j) comparator pairs.
                                              Indices i and j refer to positions in the sequence.

    Returns:
        List[int]: Output list after applying all comparators.
    """
    output = list(input_sequence) # Work on a mutable copy
    for i, j in comparators:
        # Ensure indices are valid
        if not (0 <= i < len(output) and 0 <= j < len(output)):
            logger.warning(f"Invalid comparator indices ({i}, {j}) for sequence length {len(output)}")
            continue # Or raise an error depending on desired strictness
        if output[i] > output[j]:
            output[i], output[j] = output[j], output[i]
    return output

def is_sorted(sequence: Sequence[int]) -> bool:
    """Checks if a sequence of numbers is sorted in ascending order.

    Args:
        sequence (Sequence[int]): The sequence (list or tuple) to check.

    Returns:
        bool: True if the sequence is sorted, False otherwise.
    """
    return all(sequence[i] <= sequence[i+1] for i in range(len(sequence)-1))

def is_sorting_network(n_wires: int, comparators: List[Tuple[int, int]]) -> bool:
    """Verifies if a comparator sequence forms a valid sorting network.

    Uses the zero-one principle: checks if the network correctly sorts all
    2^n possible binary input sequences of length n_wires.

    Args:
        n_wires (int): The number of inputs/outputs (wires) the network should sort.
        comparators (List[Tuple[int, int]]): The sequence of comparators defining the network.

    Returns:
        bool: True if it is a valid sorting network for n_wires inputs, False otherwise.
    """
    if n_wires <= 1:
        return True # Networks with 0 or 1 wire are trivially sorted

    for input_bits in itertools.product([0, 1], repeat=n_wires):
        output = apply_comparators(input_bits, comparators)
        if not is_sorted(output):
            # logger.debug(f"Network failed for input {input_bits}, output {output}") # Optional debug log
            return False
    return True

def count_sorted_inputs(n_wires: int, comparators: List[Tuple[int, int]]) -> int:
    """Counts how many of the 2^n binary inputs are sorted correctly by the network.

    Args:
        n_wires (int): Number of input wires.
        comparators (List[Tuple[int, int]]): The comparator network.

    Returns:
        int: The number of binary inputs (out of 2^n) that are correctly sorted.
    """
    if n_wires <= 1:
        return 2**n_wires # All inputs are sorted for n=0 or n=1

    count = 0
    for input_bits in itertools.product([0, 1], repeat=n_wires):
        output = apply_comparators(input_bits, comparators)
        if is_sorted(output):
            count += 1
    return count

def prune_redundant_comparators(n_wires: int, comparators: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Removes redundant comparators from a *valid* sorting network.

    Iterates through the comparators (usually backwards) and removes one if
    the network remains a valid sorting network without it.

    Args:
        n_wires (int): Number of wires the network sorts.
        comparators (List[Tuple[int, int]]): A *valid* sorting network sequence.

    Returns:
        List[Tuple[int, int]]: A potentially shorter list of comparators
                               that still forms a valid sorting network. Returns
                               the original list if no comparators could be pruned.
    """
    if not is_sorting_network(n_wires, comparators):
        logger.warning("Pruning called on an invalid sorting network. Returning original.")
        return comparators.copy()

    pruned_network = comparators.copy()
    i = len(pruned_network) - 1
    while i >= 0:
        # Temporarily remove the comparator at index i
        comp_to_test = pruned_network.pop(i)

        # Check if the network is still valid without this comparator
        if not is_sorting_network(n_wires, pruned_network):
            # If not valid, put the comparator back in its original position
            pruned_network.insert(i, comp_to_test)
            # logger.debug(f"Comparator {comp_to_test} at index {i} is necessary.") # Optional

        # Move to the previous comparator index
        i -= 1

    return pruned_network


def format_network_visualization(comparators: List[Tuple[int, int]], n_wires: int) -> str:
    """Formats a text-based visualization of the sorting network.

    Args:
        comparators (List[Tuple[int, int]]): The sequence of comparators.
        n_wires (int): The number of wires in the network.

    Returns:
        str: A multi-line string representing the network visually.
             Returns an empty string if comparators list is empty.
    """
    if not comparators:
        return "Empty network."

    num_comparators = len(comparators)
    # Initialize grid with horizontal lines
    grid = [['─'] * num_comparators for _ in range(n_wires)]

    for t, comp in enumerate(comparators):
         # Ensure canonical order for visualization consistency if needed, though not strictly required here
        i, j = min(comp), max(comp)

        # Check bounds
        if not (0 <= i < n_wires and 0 <= j < n_wires):
            logger.warning(f"Skipping visualization for invalid comparator {comp} at step {t}")
            continue

        # Place circles at the ends of the comparator
        if grid[i][t] == '─': grid[i][t] = "●"
        if grid[j][t] == '─': grid[j][t] = "●"

        # Place vertical lines for wires between i and j at this step
        for wire_idx in range(i + 1, j):
             if grid[wire_idx][t] == '─': # Avoid overwriting circles if comparators overlap vertically (unlikely in simple generation)
                grid[wire_idx][t] = "|"

    # Format the output string
    lines = []
    for wire in range(n_wires):
        lines.append(f"w{wire}: " + " ".join(grid[wire]))
    return "\n".join(lines)