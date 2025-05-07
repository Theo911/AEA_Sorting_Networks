"""
Performance Analysis for Batcher's Odd-Even Mergesort algorithm.

This module provides tools to analyze the performance of the Batcher's 
Odd-Even Mergesort algorithm, including:
- Comparator count across different input sizes
- Depth (parallel steps) analysis
- Comparison with known optimal values
"""

import numpy as np
from typing import List, Tuple, Dict
import time

from .core import generate_sorting_network
from .network_properties import find_parallel_layers


def analyze_comparator_count(min_size: int = 2, max_size: int = 32) -> Dict[int, int]:
    """
    Count the number of comparators needed for different input sizes.
    
    Args:
        min_size: Smallest input size to analyze
        max_size: Largest input size to analyze
        
    Returns:
        Dictionary mapping input size to comparator count
    """
    sizes = range(min_size, max_size + 1)
    results = {}
    
    for n in sizes:
        comparators = generate_sorting_network(n)
        results[n] = len(comparators)
    
    return results


def count_depth(comparators: List[Tuple[int, int]], n_wires: int) -> int:
    """
    Count the depth (number of parallel steps) of a sorting network.
    
    Args:
        comparators: List of (i,j) comparator pairs
        n_wires: Number of wires
        
    Returns:
        Depth of the network
    """
    # Use find_parallel_layers from network_properties.py
    layers = find_parallel_layers(comparators, n_wires)
    return len(layers)


def analyze_network_depth(min_size: int = 2, max_size: int = 32) -> Dict[int, int]:
    """
    Analyze the depth of networks for different input sizes.
    
    Args:
        min_size: Smallest input size to analyze
        max_size: Largest input size to analyze
        
    Returns:
        Dictionary mapping input size to network depth
    """
    sizes = range(min_size, max_size + 1)
    results = {}
    
    for n in sizes:
        comparators = generate_sorting_network(n)
        depth = count_depth(comparators, n)
        results[n] = depth
    
    return results


def timing_analysis(min_size: int = 2, max_size: int = 20, trials: int = 5) -> Dict[int, float]:
    """
    Measure the time taken to generate networks of different sizes.
    
    Args:
        min_size: Smallest input size to test
        max_size: Largest input size to test
        trials: Number of trials for averaging
        
    Returns:
        Dictionary mapping input size to average generation time (ms)
    """
    sizes = range(min_size, max_size + 1)
    results = {}
    
    for n in sizes:
        total_time = 0
        
        for _ in range(trials):
            start_time = time.time()
            generate_sorting_network(n)
            total_time += (time.time() - start_time)
        
        avg_time = (total_time / trials) * 1000  # Convert to milliseconds
        results[n] = avg_time
    
    return results


def compare_with_optimal() -> Dict[str, Dict[int, int]]:
    """
    Compare Batcher's algorithm with known optimal values.
    
    Returns:
        Dictionary with comparisons for sizes and depths
    """
    # Known optimal values (size and depth) for small n
    # Source: Knuth's "The Art of Computer Programming" and research papers
    optimal_sizes = {
        1: 0, 2: 1, 3: 3, 4: 5, 5: 9, 6: 12, 7: 16, 8: 19,
        9: 25, 10: 29, 11: 35, 12: 39, 13: 45, 14: 51, 15: 56, 16: 60
    }
    
    optimal_depths = {
        1: 0, 2: 1, 3: 3, 4: 3, 5: 5, 6: 5, 7: 6, 8: 6,
        9: 7, 10: 7, 11: 8, 12: 8, 13: 9, 14: 9, 15: 9, 16: 9
    }
    
    # Get Batcher's values
    batcher_sizes = analyze_comparator_count(1, 16)
    batcher_depths = analyze_network_depth(1, 16)
    
    return {
        "optimal_sizes": optimal_sizes,
        "batcher_sizes": batcher_sizes,
        "optimal_depths": optimal_depths,
        "batcher_depths": batcher_depths
    } 