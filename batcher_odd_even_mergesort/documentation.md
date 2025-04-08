# Batcher's Odd-Even Mergesort Algorithm Documentation

## Overview

Batcher's Odd-Even Mergesort is a sorting network construction algorithm developed by Kenneth E. Batcher in 1968. The algorithm constructs an efficient sorting network through a recursive divide-and-conquer approach. Unlike traditional sorting algorithms like quicksort or mergesort that make data-dependent decisions, sorting networks perform a fixed sequence of comparisons regardless of the input data.

## Algorithm Description

The algorithm consists of three key components:

1. **Odd-Even Merge Sort**: The main function that recursively divides input into halves, sorts them, and then merges.
2. **Odd-Even Merge**: Merges two sorted sequences through a recursive process.
3. **Odd-Even Merge Compare**: Creates the actual comparator operations for comparing elements.

### Pseudo-code

```
function odd_even_merge_sort(lo, hi):
    if hi - lo <= 1:
        return []  # Base case: single element is already sorted
    
    middle = (lo + hi) / 2
    comparators = []
    
    # Recursively sort first half
    comparators.extend(odd_even_merge_sort(lo, middle))
    
    # Recursively sort second half
    comparators.extend(odd_even_merge_sort(middle, hi))
    
    # Merge the two sorted halves
    comparators.extend(odd_even_merge(lo, hi, 1))
    
    return comparators

function odd_even_merge(lo, hi, r):
    if hi - lo <= 1:
        return []
    
    middle = (lo + hi) / 2
    comparators = []
    
    # Recursively merge first and second half
    comparators.extend(odd_even_merge(lo, middle, r))
    comparators.extend(odd_even_merge(middle, hi, r))
    
    # Add compare-exchange operations
    comparators.extend(odd_even_merge_compare(lo, hi, r))
    
    return comparators

function odd_even_merge_compare(lo, hi, r):
    comparators = []
    d = r * 2
    
    if d < hi - lo:
        for i = lo + r to hi - r with step d:
            comparators.append((i, i + r))
    
    return comparators
```

## Mathematical Properties

### Complexity Analysis

- **Comparator Count**: O(n log²n)
  - For n inputs, Batcher's network uses approximately (n/2)log²n comparators
  - This is higher than the theoretical lower bound of O(n log n)

- **Depth (Parallel Steps)**: O(log²n)
  - The algorithm allows parallel execution of comparators
  - Depth represents the number of steps required if all independent comparators are executed simultaneously

### Zero-One Principle

Batcher's network satisfies the zero-one principle, which states that:

> If a sorting network correctly sorts all 2^n possible binary sequences (with only 0s and 1s), then it will correctly sort any sequence of arbitrary values.

This principle is fundamental in proving the correctness of sorting networks and significantly simplifies verification.

## Implementation Details

Our implementation consists of several Python modules:

1. **batcher_odd_even_mergesort.py**: Core implementation of the algorithm
2. **visualization.py**: Tools for visualizing the sorting network
3. **performance_analysis.py**: Performance analysis and comparison with optimal values
4. **network_properties.py**: Analysis of mathematical properties

### Core Functions

```python
def generate_sorting_network(n: int) -> List[Tuple[int, int]]:
    """Generate a sorting network for n inputs."""
    return odd_even_merge_sort(0, n)

def apply_comparators(input_list: List[int], comparators: List[Tuple[int, int]]) -> List[int]:
    """Apply the sorting network to an input list."""
    output = input_list.copy()
    for i, j in comparators:
        if output[i] > output[j]:
            output[i], output[j] = output[j], output[i]
    return output
```

## Visual Representation

Sorting networks are typically visualized as horizontal wires with vertical comparators. Each comparator connects two wires and performs a comparison-exchange operation:

```
Wire 0 ───────●──────────●───────────────●─────
              │            │               │
Wire 1 ───────●──────────●───────────────●─────
                          │               │
Wire 2 ─────────●─────────●──────────●────────
                │                     │
Wire 3 ─────────●─────────────────────●────────
```

In this diagram:
- Horizontal lines represent wires carrying values
- Vertical lines with dots represent comparators
- Each comparator compares values on two wires and swaps them if needed

## Performance Analysis

### Comparator Count for Various Input Sizes

| Input Size | Comparators | Optimal | Overhead |
|------------|-------------|---------|----------|
| 4          | 5           | 5       | 0%       |
| 8          | 19          | 19      | 0%       |
| 16         | 63          | 60      | 5%       |

### Depth Analysis

| Input Size | Depth | Optimal Depth | Difference |
|------------|-------|---------------|------------|
| 4          | 3     | 3             | 0          |
| 8          | 6     | 6             | 0          |
| 16         | 10    | 9             | 1          |

### Comparison with Other Algorithms

Compared to other sorting network algorithms:

- **Bitonic Sort**: Similar asymptotic complexity, but Bitonic sort is only optimal for powers of 2
- **Insertion Sort Networks**: Simpler but less efficient with O(n²) comparators
- **AKS Networks**: Better theoretical complexity O(n log n) but impractical constants
- **Optimal Networks**: Smaller networks (< 10 inputs) have known optimal solutions

## Practical Applications

1. **Hardware Implementation**: Fixed comparison structure is ideal for hardware circuits
2. **Parallel Processing**: Natural parallelism makes it suitable for GPU implementations
3. **Network Routers**: Used in routing and switching networks
4. **SIMD Architectures**: Efficient for single-instruction-multiple-data architectures

## Conclusion

Batcher's Odd-Even Mergesort provides a practical and efficient sorting network construction algorithm with good theoretical properties. While not optimal in terms of comparator count for all input sizes, it has several advantages:

1. Works for any input size (not just powers of 2)
2. Simple and elegant recursive structure
3. Good parallelism
4. Near-optimal performance for many input sizes

## References

1. Batcher, K. E. (1968). "Sorting networks and their applications." In *Proceedings of the AFIPS Spring Joint Computer Conference*, 307-314.

2. Knuth, D. E. (1998). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley.

3. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. 