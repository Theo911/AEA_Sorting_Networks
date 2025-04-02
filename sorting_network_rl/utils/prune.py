import itertools
from typing import List, Tuple

def apply_comparators(inputs: List[int], comparators: List[Tuple[int, int]]) -> List[int]:
    out = inputs.copy()
    for i, j in comparators:
        if out[i] > out[j]:
            out[i], out[j] = out[j], out[i]
    return out

def is_sorted(sequence: List[int]) -> bool:
    return all(x <= y for x, y in zip(sequence, sequence[1:]))

def prune_redundant_comparators(n: int, comparators: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Prune redundant comparators from a sorting network.
    Args:
        n (int): Number of wires (channels).
        comparators (List[Tuple[int, int]]): List of comparator pairs (i, j).
    """
    inputs = list(itertools.product([0, 1], repeat=n))
    pruned = []

    for idx, (i, j) in enumerate(comparators):
        essential = False
        for x in inputs:
            before = apply_comparators(list(x), pruned)
            with_current = before.copy()
            if with_current[i] > with_current[j]:
                with_current[i], with_current[j] = with_current[j], with_current[i]
            if is_sorted(with_current) and not is_sorted(before):
                essential = True
                break
        if essential:
            pruned.append((i, j))

    return pruned
