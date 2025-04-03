'''
Batcher's Odd-Even Mergesort Algorithm for Sorting Networks

This implementation creates a sorting network using Batcher's Odd-Even Mergesort algorithm.
It is a divide-and-conquer algorithm that recursively splits the inputs, sorts them,
and then merges them using an "odd-even" merge pattern.

Key features:
- Produces a sorting network with O(n log²(n)) comparators
- Works for any input size (not just powers of 2)
- Deterministic - same comparators always produced for a given input size
'''

from typing import List, Tuple


def odd_even_merge(lo: int, hi: int, r: int) -> List[Tuple[int, int]]:
    """
    Generate comparators for the odd-even merge operation.
    
    Args:
        lo: lower bound of range
        hi: upper bound of range
        r: step size
        
    Returns:
        List of (i,j) comparator pairs
    """
    comparators = []
    
    # Base case: only one element
    if hi - lo <= 1:
        return []
        
    # Recursive case
    m = (lo + hi) // 2
    
    # Recursively merge two halves
    comparators.extend(odd_even_merge(lo, m, r))
    comparators.extend(odd_even_merge(m, hi, r))
    
    # Create odd-even merge pattern
    comparators.extend(odd_even_merge_compare(lo, hi, r))
    
    return comparators


def odd_even_merge_compare(lo: int, hi: int, r: int) -> List[Tuple[int, int]]:
    """
    Generate comparators for comparing elements within an odd-even merge.
    
    Args:
        lo: lower bound of range
        hi: upper bound of range
        r: step size
        
    Returns:
        List of (i,j) comparator pairs
    """
    comparators = []
    
    # Compare elements that are 'r' distance apart
    d = r * 2
    if d < hi - lo:
        for i in range(lo + r, hi - r, d):
            comparators.append((i, i + r))
            
    return comparators


def odd_even_merge_sort(lo: int, hi: int) -> List[Tuple[int, int]]:
    """
    Generate comparators for the odd-even merge sort.
    
    Args:
        lo: lower bound of range (inclusive)
        hi: upper bound of range (exclusive)
        
    Returns:
        List of (i,j) comparator pairs that form a sorting network
    """
    comparators = []
    
    # Base case: only one element
    if hi - lo <= 1:
        return []
    
    # Recursive case: split into two halves, sort them, then merge
    m = (lo + hi) // 2
    
    # Sort first half and second half recursively
    comparators.extend(odd_even_merge_sort(lo, m))
    comparators.extend(odd_even_merge_sort(m, hi))
    
    # Merge the two sorted halves
    comparators.extend(odd_even_merge(lo, hi, 1))
    
    return comparators


def generate_sorting_network(n: int) -> List[Tuple[int, int]]:
    """
    Generate a Batcher's odd-even merge sorting network for n inputs.
    
    Args:
        n: Number of inputs to sort
        
    Returns:
        List of (i,j) comparator pairs forming a sorting network
    """
    return odd_even_merge_sort(0, n)


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
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))


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
    import itertools
    
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