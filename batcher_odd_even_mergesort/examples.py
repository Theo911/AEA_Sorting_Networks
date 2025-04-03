"""
Examples demonstrating the Batcher's Odd-Even Mergesort sorting network.

This module provides detailed examples of how the algorithm works,
along with step-by-step visualizations of the process.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List, Tuple

from batcher_odd_even_mergesort import generate_sorting_network, apply_comparators
from visualization import draw_network, visualize_network_execution, draw_depth_layers
from network_properties import print_network_analysis, analyze_layer_properties

# Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def trace_comparator_generation(n: int) -> None:
    """
    Trace the process of generating comparators in Batcher's algorithm.
    
    Args:
        n: Number of inputs
    """
    print(f"\nTracing comparator generation for n={n}:")
    print("=" * 50)
    
    comparators = []
    
    # Recursive function to trace odd-even mergesort
    def trace_merge(lo: int, hi: int, depth: int = 0) -> None:
        indent = "  " * depth
        print(f"{indent}Merge range: [{lo}, {hi})")
        
        if hi - lo <= 1:
            print(f"{indent}Base case: range size <= 1, nothing to do")
            return
        
        mid = lo + ((hi - lo) // 2)
        print(f"{indent}Split into: [{lo}, {mid}) and [{mid}, {hi})")
        
        # Sort each half recursively
        print(f"{indent}Recursively sort first half: [{lo}, {mid})")
        trace_merge(lo, mid, depth + 1)
        
        print(f"{indent}Recursively sort second half: [{mid}, {hi})")
        trace_merge(mid, hi, depth + 1)
        
        # Merge the sorted halves
        print(f"{indent}Merge sorted halves: [{lo}, {mid}) and [{mid}, {hi})")
        
        if hi - lo == 2:
            # Only one comparison needed
            print(f"{indent}  Add comparator: ({lo}, {mid})")
            comparators.append((lo, mid))
        else:
            # Add odd-even merge comparators
            print(f"{indent}  Add odd-even merge comparators:")
            
            # Handle odd indices from each half
            print(f"{indent}    Odd indices merge:")
            trace_odd_even_merge(lo, hi, 1, depth + 2)
            
            # Handle even indices from each half
            print(f"{indent}    Even indices merge:")
            trace_odd_even_merge(lo, hi, 0, depth + 2)
            
            # Final step: adjacent comparators
            print(f"{indent}  Final adjacent comparators:")
            for i in range(lo + 1, hi - 1, 2):
                print(f"{indent}    Add comparator: ({i}, {i+1})")
                comparators.append((i, i+1))
    
    def trace_odd_even_merge(lo: int, hi: int, start: int, depth: int = 0) -> None:
        indent = "  " * depth
        length = hi - lo
        
        if length > 2:
            print(f"{indent}Sub-merge range: [{lo+start}, {hi}) with step=2")
            for i in range(lo + start, mid, 2):
                j = i + length // 2
                print(f"{indent}  Add comparator: ({i}, {j})")
                comparators.append((i, j))
        
    # Define mid for tracing
    mid = n // 2
    
    # Start the tracing process
    trace_merge(0, n)
    
    print("\nGenerated comparators:")
    for i, comp in enumerate(comparators):
        print(f"{i+1}: {comp}")
    
    # Save trace to file
    with open(os.path.join(RESULTS_DIR, f'comparator_trace_n{n}.md'), 'w') as f:
        f.write(f"# Trace of Comparator Generation for n={n}\n\n")
        f.write("```\n")
        f.write(f"Tracing comparator generation for n={n}:\n")
        f.write("=" * 50 + "\n")
        
        # Reset and rerun to capture output
        comparators.clear()
        
        # Redirect print statements to file
        original_print = print
        outputs = []
        
        def file_print(*args, **kwargs):
            output = " ".join(str(arg) for arg in args)
            outputs.append(output)
        
        # Replace print function temporarily
        print_func = file_print
        
        # Re-run the trace with modified print function
        def trace_with_capture():
            nonlocal print_func
            
            def print_capture(*args, **kwargs):
                return print_func(*args, **kwargs)
            
            # Track original print function
            old_print = print_func
            
            # Run the trace with our capturing print function
            def _trace_merge(lo: int, hi: int, depth: int = 0):
                indent = "  " * depth
                print_capture(f"{indent}Merge range: [{lo}, {hi})")
                
                if hi - lo <= 1:
                    print_capture(f"{indent}Base case: range size <= 1, nothing to do")
                    return
                
                mid = lo + ((hi - lo) // 2)
                print_capture(f"{indent}Split into: [{lo}, {mid}) and [{mid}, {hi})")
                
                print_capture(f"{indent}Recursively sort first half: [{lo}, {mid})")
                _trace_merge(lo, mid, depth + 1)
                
                print_capture(f"{indent}Recursively sort second half: [{mid}, {hi})")
                _trace_merge(mid, hi, depth + 1)
                
                print_capture(f"{indent}Merge sorted halves: [{lo}, {mid}) and [{mid}, {hi})")
                
                if hi - lo == 2:
                    print_capture(f"{indent}  Add comparator: ({lo}, {mid})")
                    comparators.append((lo, mid))
                else:
                    print_capture(f"{indent}  Add odd-even merge comparators:")
                    
                    print_capture(f"{indent}    Odd indices merge:")
                    _trace_odd_even_merge(lo, hi, 1, depth + 2)
                    
                    print_capture(f"{indent}    Even indices merge:")
                    _trace_odd_even_merge(lo, hi, 0, depth + 2)
                    
                    print_capture(f"{indent}  Final adjacent comparators:")
                    for i in range(lo + 1, hi - 1, 2):
                        print_capture(f"{indent}    Add comparator: ({i}, {i+1})")
                        comparators.append((i, i+1))
            
            def _trace_odd_even_merge(lo: int, hi: int, start: int, depth: int = 0):
                indent = "  " * depth
                length = hi - lo
                
                if length > 2:
                    print_capture(f"{indent}Sub-merge range: [{lo+start}, {hi}) with step=2")
                    for i in range(lo + start, mid, 2):
                        j = i + length // 2
                        print_capture(f"{indent}  Add comparator: ({i}, {j})")
                        comparators.append((i, j))
            
            _trace_merge(0, n)
            
            # Restore print function
            print_func = old_print
        
        # Run the trace with output capture
        trace_with_capture()
        
        # Write outputs to file
        for output in outputs:
            f.write(output + "\n")
        
        f.write("\nGenerated comparators:\n")
        for i, comp in enumerate(comparators):
            f.write(f"{i+1}: {comp}\n")
        
        f.write("```\n")
    
    # Draw the network
    plt.figure(figsize=(10, 6))
    draw_network(comparators, n)
    plt.savefig(os.path.join(RESULTS_DIR, f'traced_network_n{n}.png'))


def show_example_execution(n: int, input_values: List[int] = None) -> None:
    """
    Show a detailed example of executing the sorting network on a given input.
    
    Args:
        n: Number of inputs
        input_values: Input values (random if not provided)
    """
    if input_values is None:
        input_values = np.random.randint(1, 100, n).tolist()
    
    print(f"\nExample execution for n={n}:")
    print("=" * 50)
    print(f"Input values: {input_values}")
    
    comparators = generate_sorting_network(n)
    print(f"Number of comparators: {len(comparators)}")
    
    # Step-by-step execution
    print("\nStep-by-step execution:")
    current = input_values.copy()
    
    for i, (a, b) in enumerate(comparators):
        before = current.copy()
        
        # Apply comparator
        if current[a] > current[b]:
            current[a], current[b] = current[b], current[a]
            swap = "yes"
        else:
            swap = "no"
        
        print(f"Step {i+1}: Comparing ({a}, {b}) - {before[a]} vs {before[b]} - Swap: {swap}")
        print(f"   Values: {current}")
    
    print(f"\nFinal sorted output: {current}")
    print(f"Correctly sorted: {current == sorted(input_values)}")
    
    # Save execution to file
    with open(os.path.join(RESULTS_DIR, f'execution_example_n{n}.md'), 'w') as f:
        f.write(f"# Example Execution for n={n}\n\n")
        f.write(f"Input values: {input_values}\n\n")
        f.write(f"Number of comparators: {len(comparators)}\n\n")
        
        f.write("## Step-by-step execution\n\n")
        f.write("| Step | Comparator | Values Before | Comparison | Swap | Values After |\n")
        f.write("|------|------------|--------------|------------|------|-------------|\n")
        
        current = input_values.copy()
        
        for i, (a, b) in enumerate(comparators):
            before = current.copy()
            
            # Apply comparator
            if current[a] > current[b]:
                current[a], current[b] = current[b], current[a]
                swap = "Yes"
            else:
                swap = "No"
            
            before_str = str(before)
            current_str = str(current)
            comparison = f"{before[a]} vs {before[b]}"
            
            f.write(f"| {i+1} | ({a}, {b}) | {before_str} | {comparison} | {swap} | {current_str} |\n")
        
        f.write(f"\nFinal sorted output: {current}\n\n")
        f.write(f"Correctly sorted: {current == sorted(input_values)}\n")
    
    # Visualize the execution
    plt.figure(figsize=(12, 8))
    visualize_network_execution(comparators, input_values)
    plt.savefig(os.path.join(RESULTS_DIR, f'execution_visual_n{n}.png'))


def show_property_examples(n: int) -> None:
    """
    Demonstrate mathematical properties of the sorting network.
    
    Args:
        n: Number of inputs
    """
    print(f"\nDemonstrating properties for n={n}:")
    print("=" * 50)
    
    comparators = generate_sorting_network(n)
    
    # Show depth layers
    print("Parallel execution layers:")
    layer_data = analyze_layer_properties(comparators, n)
    
    for i, layer in enumerate(layer_data["layers"]):
        print(f"Layer {i}: {layer}")
    
    print(f"\nNetwork depth: {layer_data['total_layers']} layers")
    print(f"Theoretical depth lower bound: {int(np.log2(n) * (np.log2(n) + 1) / 2)}")
    
    # Draw depth layers
    plt.figure(figsize=(12, 8))
    draw_depth_layers(comparators, n)
    plt.savefig(os.path.join(RESULTS_DIR, f'depth_layers_n{n}.png'))
    
    # Verify properties and save to file
    print_network_analysis(n)


def explain_algorithm() -> None:
    """
    Provide a detailed explanation of how Batcher's odd-even mergesort works.
    """
    explanation = """
# Batcher's Odd-Even Mergesort Algorithm Explanation

Batcher's odd-even mergesort is a sorting network construction algorithm that builds on the principles of merge sort but adapts it to work in a sorting network context. Here's how it works:

## 1. Divide and Conquer Approach

Like merge sort, Batcher's algorithm uses a divide-and-conquer approach:
- Split the input sequence into two halves
- Sort each half recursively
- Merge the sorted halves

The key insight is how to implement the "merge" step using a fixed pattern of comparators.

## 2. Odd-Even Merge

The merge operation is implemented using an "odd-even merge" technique:
- It first handles odd-indexed elements from each half
- Then handles even-indexed elements from each half
- Finally adds "cleanup" comparators between adjacent elements

## 3. Key Properties

- **Network Depth**: O(log²(n))
- **Comparator Count**: O(n·log²(n))
- **Parallelism**: Multiple comparisons can happen simultaneously
- **Zero-One Principle**: If it sorts all binary sequences correctly, it will sort any sequence correctly

## 4. Recursive Implementation

The algorithm can be described recursively:
- Base case: A single element is already sorted
- For n>1:
  1. Sort first half
  2. Sort second half
  3. Apply odd-even merge to combine the results

## 5. Comparison to Other Networks

Compared to other sorting networks:
- More efficient than bubble sort's O(n²) comparators
- Though not optimal, it's practical and has a regular structure
- Works for inputs of any size (not just powers of 2)

## 6. Applications

Particularly useful in:
- Parallel computing architectures
- Hardware implementations (FPGA, ASIC)
- Situations where fixed sorting patterns are required
"""

    # Save explanation to file
    with open(os.path.join(RESULTS_DIR, 'algorithm_explanation.md'), 'w') as f:
        f.write(explanation)
    
    print("Detailed algorithm explanation saved to 'algorithm_explanation.md'")


if __name__ == "__main__":
    print("Running examples for Batcher's Odd-Even Mergesort sorting networks...")
    
    # Example 1: Trace comparator generation
    trace_comparator_generation(8)
    
    # Example 2: Show execution example
    show_example_execution(8, [9, 5, 2, 7, 1, 8, 4, 6])
    
    # Example 3: Show execution with random input
    show_example_execution(16)
    
    # Example 4: Demonstrate network properties
    show_property_examples(8)
    
    # Example 5: Detailed algorithm explanation
    explain_algorithm()
    
    print("\nAll examples complete. Output files generated.")
    
    plt.show() 