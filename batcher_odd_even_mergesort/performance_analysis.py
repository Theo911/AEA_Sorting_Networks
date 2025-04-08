"""
Performance Analysis for Batcher's Odd-Even Mergesort algorithm.

This module provides tools to analyze the performance of the Batcher's 
Odd-Even Mergesort algorithm, including:
- Comparator count across different input sizes
- Depth (parallel steps) analysis
- Comparison with known optimal values
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import time
import os

from batcher_odd_even_mergesort import generate_sorting_network

# Create results directory if it doesn't exist
RESULTS_DIR = "results_performance_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)


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
    wire_usage = [-1] * n_wires  # Last layer where each wire was used
    max_depth = 0
    
    for i, j in comparators:
        # Find earliest layer where this comparator can be placed
        depth = max(wire_usage[i], wire_usage[j]) + 1
        max_depth = max(max_depth, depth)
        
        # Update wire usage
        wire_usage[i] = depth
        wire_usage[j] = depth
    
    return max_depth + 1  # +1 to convert from 0-indexed to count


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


def plot_comparator_count(results: Dict[int, int]) -> None:
    """
    Plot the number of comparators against input size.
    
    Args:
        results: Dictionary mapping input size to comparator count
    """
    sizes = list(results.keys())
    counts = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, counts, 'o-', linewidth=2, markersize=8)
    
    # Add trend line (n log²n)
    x = np.array(sizes)
    y = 0.5 * x * np.log2(x)**2  # Theoretical complexity: ~(n/2)log²(n)
    plt.plot(x, y, 'r--', label=r'$\frac{n}{2}\log_2^2(n)$')
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Comparators')
    plt.title("Batcher's Odd-Even Mergesort: Comparator Count vs Input Size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparator_count.png'))


def plot_depth_analysis(results: Dict[int, int]) -> None:
    """
    Plot the network depth against input size.
    
    Args:
        results: Dictionary mapping input size to network depth
    """
    sizes = list(results.keys())
    depths = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, depths, 'o-', linewidth=2, markersize=8)
    
    # Add trend line (log²n)
    x = np.array(sizes)
    y = np.log2(x)**2  # Theoretical complexity: ~log²(n)
    plt.plot(x, y, 'r--', label=r'$\log_2^2(n)$')
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Depth (Parallel Steps)')
    plt.title("Batcher's Odd-Even Mergesort: Depth vs Input Size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'depth_analysis.png'))


def plot_comparison_with_optimal(comparison_data: Dict[str, Dict[int, int]]) -> None:
    """
    Plot comparison between Batcher's algorithm and optimal values.
    
    Args:
        comparison_data: Dictionary with comparison data
    """
    # Size comparison
    optimal_sizes = comparison_data["optimal_sizes"]
    batcher_sizes = comparison_data["batcher_sizes"]
    
    common_sizes = sorted(set(optimal_sizes.keys()) & set(batcher_sizes.keys()))
    
    opt_size_values = [optimal_sizes[n] for n in common_sizes]
    bat_size_values = [batcher_sizes[n] for n in common_sizes]
    
    # Depth comparison
    optimal_depths = comparison_data["optimal_depths"]
    batcher_depths = comparison_data["batcher_depths"]
    
    opt_depth_values = [optimal_depths[n] for n in common_sizes]
    bat_depth_values = [batcher_depths[n] for n in common_sizes]
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Size comparison plot
    ax1.plot(common_sizes, opt_size_values, 'o-', label='Optimal')
    ax1.plot(common_sizes, bat_size_values, 's-', label="Batcher's")
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Number of Comparators')
    ax1.set_title('Comparator Count: Batcher vs Optimal')
    ax1.grid(True)
    ax1.legend()
    
    # Depth comparison plot
    ax2.plot(common_sizes, opt_depth_values, 'o-', label='Optimal')
    ax2.plot(common_sizes, bat_depth_values, 's-', label="Batcher's")
    ax2.set_xlabel('Input Size (n)')
    ax2.set_ylabel('Depth (Parallel Steps)')
    ax2.set_title('Network Depth: Batcher vs Optimal')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'batcher_vs_optimal.png'))


def generate_comparison_table(comparison_data: Dict[str, Dict[int, int]]) -> None:
    """
    Generate a comparison table between Batcher's and optimal values.
    
    Args:
        comparison_data: Dictionary with comparison data
    """
    optimal_sizes = comparison_data["optimal_sizes"]
    batcher_sizes = comparison_data["batcher_sizes"]
    optimal_depths = comparison_data["optimal_depths"]
    batcher_depths = comparison_data["batcher_depths"]
    
    common_sizes = sorted(set(optimal_sizes.keys()) & set(batcher_sizes.keys()))
    
    # Print table header
    print("\nComparison of Batcher's Algorithm with Known Optimal Values:\n")
    print("| Input Size | Optimal Size | Batcher Size | Overhead | Optimal Depth | Batcher Depth | Depth Diff |")
    print("|------------|--------------|--------------|----------|---------------|---------------|------------|")
    
    # Print table rows
    for n in common_sizes:
        opt_size = optimal_sizes[n]
        bat_size = batcher_sizes[n]
        size_overhead = ((bat_size - opt_size) / opt_size * 100) if opt_size > 0 else 0
        
        opt_depth = optimal_depths[n]
        bat_depth = batcher_depths[n]
        depth_diff = bat_depth - opt_depth
        
        print(f"| {n:<10} | {opt_size:<12} | {bat_size:<12} | {size_overhead:>7.1f}% | {opt_depth:<13} | {bat_depth:<13} | {depth_diff:<10} |")
    
    # Save to file
    with open(os.path.join(RESULTS_DIR, 'comparison_table.md'), 'w') as f:
        f.write("# Comparison of Batcher's Algorithm with Known Optimal Values\n\n")
        f.write("| Input Size | Optimal Size | Batcher Size | Overhead | Optimal Depth | Batcher Depth | Depth Diff |\n")
        f.write("|------------|--------------|--------------|----------|---------------|---------------|------------|\n")
        
        for n in common_sizes:
            opt_size = optimal_sizes[n]
            bat_size = batcher_sizes[n]
            size_overhead = ((bat_size - opt_size) / opt_size * 100) if opt_size > 0 else 0
            
            opt_depth = optimal_depths[n]
            bat_depth = batcher_depths[n]
            depth_diff = bat_depth - opt_depth
            
            f.write(f"| {n} | {opt_size} | {bat_size} | {size_overhead:.1f}% | {opt_depth} | {bat_depth} | {depth_diff} |\n")


if __name__ == "__main__":
    print("Analyzing Batcher's Odd-Even Mergesort performance...")
    
    # Analyze comparator count
    print("\nAnalyzing comparator count...")
    comp_results = analyze_comparator_count(2, 24)
    plot_comparator_count(comp_results)
    
    # Analyze network depth
    print("\nAnalyzing network depth...")
    depth_results = analyze_network_depth(2, 24)
    plot_depth_analysis(depth_results)
    
    # Compare with optimal values
    print("\nComparing with known optimal values...")
    comparison = compare_with_optimal()
    plot_comparison_with_optimal(comparison)
    generate_comparison_table(comparison)
    
    # Timing analysis
    print("\nPerforming timing analysis...")
    timing_results = timing_analysis(2, 20)
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(timing_results.keys()), list(timing_results.values()), 'o-')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Generation Time (ms)')
    plt.title("Generation Time for Batcher's Network")
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visualization
    plt.savefig(os.path.join(RESULTS_DIR, 'generation_time.png'))
    
    print("\nAnalysis complete. Results saved to output files.")
    
    plt.show() 