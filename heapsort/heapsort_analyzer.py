import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class HeapSortAnalyzer:
    """
    Complete implementation and analysis of Heapsort algorithm
    Includes comparison with other sorting algorithms
    """
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        
    def reset_counters(self):
        """Reset operation counters for analysis"""
        self.comparisons = 0
        self.swaps = 0
    
    def heapsort(self, arr: List[int]) -> List[int]:
        """
        Main heapsort algorithm
        
        Steps:
        1. Build max heap from input array
        2. Extract maximum elements one by one
        3. Place them at the end of array
        
        Parameters:
            arr: List of numbers to sort
            
        Returns:
            Sorted list
        """
        if len(arr) <= 1:
            return arr
            
        # Step 1: Build max heap
        self._build_max_heap(arr)
        
        # Step 2: Extract elements from heap one by one
        heap_size = len(arr)
        
        for i in range(len(arr) - 1, 0, -1):
            # Move current root (maximum) to end
            arr[0], arr[i] = arr[i], arr[0]
            self.swaps += 1
            
            # Reduce heap size and fix heap property
            heap_size -= 1
            self._max_heapify(arr, 0, heap_size)
            
        return arr
    
    def _build_max_heap(self, arr: List[int]):
        """
        Build max heap from unsorted array
        Start from last non-leaf node and heapify all nodes
        
        Parameters:
            arr: Array to convert to max heap
        """
        # Last non-leaf node is at index (n//2 - 1)
        start_idx = len(arr) // 2 - 1
        
        # Heapify all nodes from last non-leaf to root
        for i in range(start_idx, -1, -1):
            self._max_heapify(arr, i, len(arr))
    
    def _max_heapify(self, arr: List[int], root_idx: int, heap_size: int):
        """
        Maintain max heap property for subtree rooted at root_idx
        
        Parameters:
            arr: Array representing heap
            root_idx: Index of root of subtree to heapify
            heap_size: Size of heap (may be less than array size)
        """
        largest = root_idx
        left_child = 2 * root_idx + 1
        right_child = 2 * root_idx + 2
        
        # Check if left child exists and is larger than root
        if left_child < heap_size:
            self.comparisons += 1
            if arr[left_child] > arr[largest]:
                largest = left_child
        
        # Check if right child exists and is larger than current largest
        if right_child < heap_size:
            self.comparisons += 1
            if arr[right_child] > arr[largest]:
                largest = right_child
        
        # If largest is not root, swap and continue heapifying
        if largest != root_idx:
            arr[root_idx], arr[largest] = arr[largest], arr[root_idx]
            self.swaps += 1
            
            # Recursively heapify the affected subtree
            self._max_heapify(arr, largest, heap_size)
    
    def merge_sort(self, arr: List[int]) -> List[int]:
        """
        Merge sort implementation for comparison
        
        Parameters:
            arr: List to sort
            
        Returns:
            Sorted list
        """
        if len(arr) <= 1:
            return arr
            
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[int], right: List[int]) -> List[int]:
        """Merge two sorted arrays"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            self.comparisons += 1
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    def quicksort(self, arr: List[int]) -> List[int]:
        """
        Randomized quicksort implementation for comparison
        
        Parameters:
            arr: List to sort
            
        Returns:
            Sorted list
        """
        if len(arr) <= 1:
            return arr
            
        # Choose random pivot
        pivot_idx = random.randint(0, len(arr) - 1)
        pivot = arr[pivot_idx]
        
        less = []
        equal = []
        greater = []
        
        for element in arr:
            self.comparisons += 1
            if element < pivot:
                less.append(element)
            elif element == pivot:
                equal.append(element)
            else:
                greater.append(element)
        
        return self.quicksort(less) + equal + self.quicksort(greater)
    
    def generate_test_arrays(self, size: int) -> dict:
        """
        Generate different types of test arrays
        
        Parameters:
            size: Size of arrays to generate
            
        Returns:
            Dictionary with different array types
        """
        return {
            'random': [random.randint(1, 1000) for _ in range(size)],
            'sorted': list(range(1, size + 1)),
            'reverse_sorted': list(range(size, 0, -1)),
            'nearly_sorted': self._generate_nearly_sorted(size),
            'duplicates': [random.randint(1, 10) for _ in range(size)]
        }
    
    def _generate_nearly_sorted(self, size: int) -> List[int]:
        """Generate nearly sorted array with few random swaps"""
        arr = list(range(1, size + 1))
        # Perform random swaps on 5% of elements
        for _ in range(size // 20):
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    def benchmark_algorithms(self, sizes: List[int], num_trials: int = 3) -> dict:
        """
        Compare heapsort with other sorting algorithms
        
        Parameters:
            sizes: List of array sizes to test
            num_trials: Number of trials per test
            
        Returns:
            Dictionary with benchmark results
        """
        algorithms = {
            'heapsort': self.heapsort,
            'merge_sort': self.merge_sort,
            'quicksort': self.quicksort
        }
        
        results = {
            'sizes': sizes,
            'algorithms': {}
        }
        
        for alg_name in algorithms:
            results['algorithms'][alg_name] = {
                'random': [], 'sorted': [], 'reverse_sorted': [], 
                'nearly_sorted': [], 'duplicates': []
            }
        
        for size in sizes:
            print(f"Testing with array size {size}...")
            
            for array_type in ['random', 'sorted', 'reverse_sorted', 'nearly_sorted', 'duplicates']:
                for alg_name, alg_func in algorithms.items():
                    times = []
                    
                    for trial in range(num_trials):
                        test_arrays = self.generate_test_arrays(size)
                        arr = test_arrays[array_type].copy()
                        
                        self.reset_counters()
                        start_time = time.perf_counter()
                        alg_func(arr.copy())
                        end_time = time.perf_counter()
                        
                        times.append(end_time - start_time)
                    
                    avg_time = np.mean(times)
                    results['algorithms'][alg_name][array_type].append(avg_time)
        
        return results
    
    def analyze_heap_operations(self):
        """
        Analyze time complexity of heap operations
        """
        print("=== HEAP OPERATIONS TIME COMPLEXITY ANALYSIS ===\n")
        
        print("Building Max Heap:")
        print("=================")
        print("- Start from last non-leaf node at index (n//2 - 1)")
        print("- Call max_heapify on each node from bottom to top")
        print("- Each heapify operation takes O(log n) time")
        print("- Total nodes to heapify: n/2")
        print("- Time complexity: O(n) - tighter analysis shows this is optimal")
        print()
        
        print("Max Heapify Operation:")
        print("=====================")
        print("- Compare root with its children")
        print("- Swap with larger child if needed")
        print("- Recursively fix the affected subtree")
        print("- Height of binary heap: log n")
        print("- Time complexity: O(log n)")
        print()
        
        print("Heapsort Algorithm:")
        print("==================")
        print("Step 1: Build max heap - O(n)")
        print("Step 2: Extract max elements:")
        print("  - Repeat n times:")
        print("    - Swap root with last element - O(1)")
        print("    - Reduce heap size - O(1)")
        print("    - Heapify root - O(log n)")
        print("  - Total for extraction: n × O(log n) = O(n log n)")
        print()
        print("Overall Time Complexity: O(n) + O(n log n) = O(n log n)")
        print("Space Complexity: O(1) - sorts in place")
        print()
        
        # Empirical validation
        sizes = [100, 500, 1000, 2000, 5000]
        operations = []
        
        for size in sizes:
            arr = [random.randint(1, 1000) for _ in range(size)]
            self.reset_counters()
            self.heapsort(arr.copy())
            
            total_ops = self.comparisons + self.swaps
            operations.append(total_ops)
            theoretical = size * np.log2(size) * 2  # Approximate constant
            
            print(f"Array size {size}: Operations = {total_ops}, "
                  f"Theory approximately {theoretical:.0f}, "
                  f"Ratio = {total_ops/theoretical:.2f}")
    
    def plot_comparison_results(self, results: dict):
        """Create graphs showing algorithm comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        sizes = results['sizes']
        array_types = ['random', 'sorted', 'reverse_sorted', 'nearly_sorted', 'duplicates']
        colors = ['blue', 'green', 'red']
        algorithms = ['heapsort', 'merge_sort', 'quicksort']
        
        for i, array_type in enumerate(array_types):
            ax = axes[i]
            
            for j, alg in enumerate(algorithms):
                times = results['algorithms'][alg][array_type]
                ax.plot(sizes, times, 'o-', color=colors[j], 
                       label=alg.replace('_', ' ').title(), linewidth=2)
            
            ax.set_xlabel('Array Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{array_type.replace("_", " ").title()} Arrays')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Hide the 6th subplot since we only have 5 array types
        axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('heapsort_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def theoretical_analysis_demo(self):
        """
        Demonstrate theoretical analysis of heapsort
        """
        print("=== HEAPSORT THEORETICAL ANALYSIS ===\n")
        
        print("Why Heapsort is O(n log n) in ALL cases:")
        print("========================================")
        print("Unlike quicksort, heapsort has guaranteed O(n log n) performance")
        print("because its operation doesn't depend on input distribution.")
        print()
        
        print("Algorithm Steps and Complexity:")
        print("==============================")
        print("1. Build Max Heap: O(n)")
        print("   - Convert unsorted array into max heap")
        print("   - Each element settles to correct position")
        print("   - Bottom-up approach is more efficient than top-down")
        print()
        print("2. Extract Maximum Elements: O(n log n)")
        print("   - Repeat n times:")
        print("     a. Remove root (maximum) - O(1)")
        print("     b. Move last element to root - O(1)")
        print("     c. Restore heap property - O(log n)")
        print("   - Total: n × O(log n) = O(n log n)")
        print()
        
        print("Why ALL cases are O(n log n):")
        print("=============================")
        print("Best Case: Array already sorted")
        print("- Still need to build heap: O(n)")
        print("- Still need to extract all elements: O(n log n)")
        print("- Total: O(n log n)")
        print()
        print("Average Case: Random array")
        print("- Build heap: O(n)")
        print("- Extract elements: O(n log n)")
        print("- Total: O(n log n)")
        print()
        print("Worst Case: Any input")
        print("- Same operations regardless of input")
        print("- Total: O(n log n)")
        print()
        
        print("Space Complexity: O(1)")
        print("======================")
        print("- Sorts in place using the input array")
        print("- Only uses constant extra space for variables")
        print("- Very memory efficient compared to merge sort")
        print()
        
        print("Advantages of Heapsort:")
        print("======================")
        print("+ Guaranteed O(n log n) time complexity")
        print("+ O(1) space complexity (in-place sorting)")
        print("+ Not affected by input distribution")
        print("+ Good for systems with memory constraints")
        print()
        
        print("Disadvantages of Heapsort:")
        print("=========================")
        print("- Not stable (doesn't preserve relative order of equal elements)")
        print("- Poor cache performance due to random memory access")
        print("- Constant factors higher than quicksort in practice")
        print("- More complex implementation than simple algorithms")


# Example usage and testing
if __name__ == "__main__":
    analyzer = HeapSortAnalyzer()
    
    # Demonstrate basic heapsort functionality
    print("=== HEAPSORT BASIC DEMONSTRATION ===")
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array}")
    
    sorted_array = analyzer.heapsort(test_array.copy())
    print(f"Sorted array: {sorted_array}")
    print()
    
    # Show theoretical analysis
    analyzer.theoretical_analysis_demo()
    
    print("\n" + "="*60)
    print("HEAP OPERATIONS ANALYSIS")
    print("="*60)
    
    # Analyze heap operations
    analyzer.analyze_heap_operations()
    
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON TESTING")
    print("="*60)
    
    # Compare with other sorting algorithms
    sizes = [100, 500, 1000, 2000, 5000]
    results = analyzer.benchmark_algorithms(sizes, num_trials=3)
    
    # Create comparison graphs
    analyzer.plot_comparison_results(results)
    
    # Print summary
    print("\nPerformance Summary:")
    print("===================")
    algorithms = ['heapsort', 'merge_sort', 'quicksort']
    
    for array_type in ['random', 'sorted', 'reverse_sorted']:
        print(f"\n{array_type.replace('_', ' ').title()} Arrays:")
        for alg in algorithms:
            avg_time = np.mean(results['algorithms'][alg][array_type])
            print(f"  {alg.replace('_', ' ').title()}: {avg_time:.6f} seconds average")