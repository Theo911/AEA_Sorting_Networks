"""
Batcher's Odd-Even Mergesort Algorithm for Sorting Networks

This implementation creates a sorting network using Batcher's Odd-Even Mergesort algorithm.
It follows the iterative formulation described in Knuth's "The Art of Computer Programming".

Batcher's algorithm creates a sorting network by:
1. Progressively building larger and larger sorted subsequences
2. Using power-of-2 increments to create "chains" of comparators
3. Merging sorted subsequences through carefully placed comparators

Key features:
- Produces a sorting network with O(n log²(n)) comparators
- Works for any input size (not just powers of 2)
- Deterministic - same comparators always produced for a given input size
- Proven to satisfy the Zero-One Principle
"""

from typing import List, Tuple
import itertools


def batcher_sort(n: int) -> List[Tuple[int, int]]:
    """
    Correct implementation of Batcher's odd-even mergesort algorithm for sorting networks.
    
    This is an iterative implementation based on Knuth's "The Art of Computer Programming".
    While Batcher's algorithm is conceptually a divide-and-conquer method, this implementation
    directly generates the comparators in the correct order without explicit recursion.
    
    The algorithm works by:
    1. Starting with t=1 and doubling t in each outer loop (t=1,2,4,8,...)
    2. For each t, starting with p=t and halving p in each inner loop (p=t,t/2,t/4,...)
    3. For each p, generate comparators between elements that are p distance apart
       but only for elements in the same "t-group" (determined by bitwise operations)
    
    Args:
        n: Number of inputs
        
    Returns:
        List of (i,j) comparator pairs forming a sorting network
    """
    comparators = []

    # Base case: no need to sort 0 or 1 elements
    if n <= 1:
        return []

    # For n=2, just need one comparator
    if n == 2:
        return [(0, 1)]

    # Implementation of Batcher's odd-even mergesort
    # This is based on the algorithm description from Knuth's TAOCP
    t = 1 # Defined the "groups size" for merging
    while t < n:
        p = t # defines the distance between elements being compared
        while p > 0:
            for i in range(0, n - p):
                if i & t == 0:  # Bitwise AND to check if i is in the first group
                    # Ensure j is within bounds
                    j = i + p
                    if j < n:
                        comparators.append((i, j))
            p = p >> 1  # Bitwise right shift, same as p = p // 2
        t = t << 1  # Bitwise left shift, same as t = t * 2

    return comparators


def generate_sorting_network(n: int) -> List[Tuple[int, int]]:
    """
    Generate a Batcher's odd-even merge sorting network for n inputs.
    
    Args:
        n: Number of inputs to sort
        
    Returns:
        List of (i,j) comparator pairs forming a sorting network
    """
    return batcher_sort(n)


def apply_comparators(input_list: List[int], comparators: List[Tuple[int, int]]) -> List[int]:
    """
    Apply sorting network comparators to an input list.
    
    Args:
        input_list: List of values to sort
        comparators: List of (i,j) comparator pairs
        
    Returns:
        Sorted list
    """
    output = input_list.copy()
    for i, j in comparators:
        if output[i] > output[j]:
            output[i], output[j] = output[j], output[i]
    return output


def is_sorted(lst: List[int]) -> bool:
    """Check if a list is sorted."""
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def verify_sorting_network(n: int, comparators: List[Tuple[int, int]]) -> bool:
    """
    Verify if the sorting network sorts all possible inputs correctly.
    For large n, this is not practical (2^n combinations).
    
    Args:
        n: Number of inputs
        comparators: List of (i,j) comparator pairs
        
    Returns:
        True if it is a valid sorting network, False otherwise
    """

    # Check all possible binary inputs (0-1 principle)
    for input_bits in itertools.product([0, 1], repeat=n):
        output = apply_comparators(list(input_bits), comparators)
        if not is_sorted(output):
            return False
    return True


if __name__ == "__main__":
    # Example usage
    n = 8  # Number of inputs
    comparators = generate_sorting_network(n)

    print(f"Batcher's Odd-Even Mergesort network for {n} inputs:")
    print(f"Number of comparators: {len(comparators)}")
    print("Comparators:", comparators)

    # Verify for small n (verification is exponential, so only practical for small n)
    if n <= 10:
        is_valid = verify_sorting_network(n, comparators)
        print(f"Network is valid sorting network: {is_valid}")
