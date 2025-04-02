import itertools
from typing import List, Tuple


def apply_comparators(input_list: List[int], comparators: List[Tuple[int, int]]) -> List[int]:
    """
    Applies a sequence of comparators to an input list.

    Args:
        input_list (List[int]): Binary input vector.
        comparators (List[Tuple[int, int]]): List of (i, j) comparator pairs.

    Returns:
        List[int]: Output list after applying all comparators.
    """
    output = input_list.copy()
    for i, j in comparators:
        if output[i] > output[j]:
            output[i], output[j] = output[j], output[i]
    return output


def is_sorted(lst: List[int]) -> bool:
    """
    Checks if a list is sorted in ascending order.

    Args:
        lst (List[int]): List of integers.

    Returns:
        bool: True if sorted, else False.
    """
    return all(x <= y for x, y in zip(lst, lst[1:]))


def is_sorting_network(n: int, comparators: List[Tuple[int, int]]) -> bool:
    """
    Verifies whether a sequence of comparators forms a valid sorting network
    for all binary inputs (using the zero-one principle).

    Args:
        n (int): Number of wires (channels).
        comparators (List[Tuple[int, int]]): List of comparators.

    Returns:
        bool: True if it is a valid sorting network, False otherwise.
    """
    for input_bits in itertools.product([0, 1], repeat=n):
        output = apply_comparators(list(input_bits), comparators)
        if not is_sorted(output):
            return False
    return True


def count_sorted_inputs(n: int, comparators: List[Tuple[int, int]]) -> int:
    """
    Counts how many of the 2^n binary inputs are sorted correctly by the given comparator network.

    Args:
        n (int): Number of input wires.
        comparators (List[Tuple[int, int]]): The comparator network.

    Returns:
        int: Number of sorted binary inputs.
    """
    count = 0
    for input_bits in itertools.product([0, 1], repeat=n):
        output = apply_comparators(list(input_bits), comparators)
        if is_sorted(output):
            count += 1
    return count