"""
Unit tests for Heapsort implementation
Tests correctness, edge cases, and performance characteristics
"""

import unittest
import sys
import os
import random
import time

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'heapsort'))

from heapsort_analyzer import HeapSortAnalyzer

class TestHeapSort(unittest.TestCase):
    """Test cases for Heapsort implementation"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.analyzer = HeapSortAnalyzer()
        
    def test_empty_array(self):
        """Test sorting empty array"""
        arr = []
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, [])
        
    def test_single_element(self):
        """Test sorting single element array"""
        arr = [42]
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, [42])
        
    def test_two_elements_sorted(self):
        """Test sorting two element array already sorted"""
        arr = [1, 2]
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, [1, 2])
        
    def test_two_elements_reverse(self):
        """Test sorting two element array in reverse order"""
        arr = [2, 1]
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, [1, 2])
        
    def test_small_array_sorted(self):
        """Test sorting small already sorted array"""
        arr = [1, 2, 3, 4, 5]
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, [1, 2, 3, 4, 5])
        
    def test_small_array_reverse_sorted(self):
        """Test sorting small reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, [1, 2, 3, 4, 5])
        
    def test_small_array_random(self):
        """Test sorting small random array"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
        result = self.analyzer.heapsort(arr.copy())
        expected = [1, 1, 2, 3, 4, 5, 5, 6, 9]
        self.assertEqual(result, expected)
        
    def test_duplicate_elements(self):
        """Test sorting array with duplicate elements"""
        arr = [5, 2, 8, 2, 9, 1, 5, 5]
        result = self.analyzer.heapsort(arr.copy())
        expected = [1, 2, 2, 5, 5, 5, 8, 9]
        self.assertEqual(result, expected)
        
    def test_all_same_elements(self):
        """Test sorting array with all identical elements"""
        arr = [7, 7, 7, 7, 7]
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, [7, 7, 7, 7, 7])
        
    def test_negative_numbers(self):
        """Test sorting array with negative numbers"""
        arr = [-3, 1, -4, 1, 5, -9, 2, 6, -5, 3]
        result = self.analyzer.heapsort(arr.copy())
        expected = [-9, -5, -4, -3, 1, 1, 2, 3, 5, 6]
        self.assertEqual(result, expected)
        
    def test_mixed_positive_negative(self):
        """Test sorting array with mixed positive and negative numbers"""
        arr = [0, -1, 1, -2, 2, -3, 3]
        result = self.analyzer.heapsort(arr.copy())
        expected = [-3, -2, -1, 0, 1, 2, 3]
        self.assertEqual(result, expected)
        
    def test_large_numbers(self):
        """Test sorting array with large numbers"""
        arr = [1000000, 999999, 1000001, 500000]
        result = self.analyzer.heapsort(arr.copy())
        expected = [500000, 999999, 1000000, 1000001]
        self.assertEqual(result, expected)
        
    def test_random_array_medium(self):
        """Test sorting medium-sized random array"""
        arr = [random.randint(1, 100) for _ in range(20)]
        original = arr.copy()
        result = self.analyzer.heapsort(arr.copy())
        
        # Check that result is sorted
        self.assertEqual(result, sorted(original))
        
    def test_random_array_large(self):
        """Test sorting larger random array"""
        arr = [random.randint(1, 1000) for _ in range(100)]
        original = arr.copy()
        result = self.analyzer.heapsort(arr.copy())
        
        # Check that result is sorted
        self.assertEqual(result, sorted(original))
        
    def test_already_sorted_large(self):
        """Test sorting large already sorted array"""
        arr = list(range(1, 101))
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, list(range(1, 101)))
        
    def test_reverse_sorted_large(self):
        """Test sorting large reverse sorted array"""
        arr = list(range(100, 0, -1))
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, list(range(1, 101)))
        
    def test_nearly_sorted_array(self):
        """Test sorting nearly sorted array"""
        arr = list(range(1, 21))  # [1, 2, 3, ..., 20]
        # Make a few random swaps
        for _ in range(3):
            i, j = random.randint(0, 19), random.randint(0, 19)
            arr[i], arr[j] = arr[j], arr[i]
        
        result = self.analyzer.heapsort(arr.copy())
        self.assertEqual(result, list(range(1, 21)))

class TestHeapOperations(unittest.TestCase):
    """Test cases for heap operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = HeapSortAnalyzer()
        
    def test_build_max_heap_simple(self):
        """Test building max heap on simple array"""
        arr = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
        self.analyzer._build_max_heap(arr)
        
        # Check max heap property: parent >= children
        self.assertTrue(self._is_max_heap(arr))
        
    def test_build_max_heap_sorted(self):
        """Test building max heap on sorted array"""
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.analyzer._build_max_heap(arr)
        
        # Check max heap property
        self.assertTrue(self._is_max_heap(arr))
        
    def test_build_max_heap_reverse_sorted(self):
        """Test building max heap on reverse sorted array"""
        arr = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        self.analyzer._build_max_heap(arr)
        
        # Check max heap property
        self.assertTrue(self._is_max_heap(arr))
        
    def test_max_heapify_internal_node(self):
        """Test max heapify operation on internal node"""
        arr = [16, 4, 10, 14, 7, 9, 3, 2, 8, 1]
        
        # Violate heap property at index 1
        self.analyzer._max_heapify(arr, 1, len(arr))
        
        # Check that subtree is now a valid heap
        self.assertTrue(self._is_subtree_max_heap(arr, 1))
        
    def test_counter_tracking(self):
        """Test that operation counters work correctly"""
        arr = [5, 3, 8, 1, 9, 2]
        
        self.analyzer.reset_counters()
        initial_comparisons = self.analyzer.comparisons
        initial_swaps = self.analyzer.swaps
        
        self.analyzer.heapsort(arr.copy())
        
        # Check that counters increased
        self.assertGreater(self.analyzer.comparisons, initial_comparisons)
        self.assertGreater(self.analyzer.swaps, initial_swaps)
        
    def _is_max_heap(self, arr):
        """Helper method to check if array satisfies max heap property"""
        n = len(arr)
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            # Check left child
            if left_child < n and arr[i] < arr[left_child]:
                return False
                
            # Check right child
            if right_child < n and arr[i] < arr[right_child]:
                return False
                
        return True
        
    def _is_subtree_max_heap(self, arr, root_index):
        """Helper method to check if subtree satisfies max heap property"""
        n = len(arr)
        if root_index >= n:
            return True
            
        left_child = 2 * root_index + 1
        right_child = 2 * root_index + 2
        
        # Check left child
        if left_child < n:
            if arr[root_index] < arr[left_child]:
                return False
            if not self._is_subtree_max_heap(arr, left_child):
                return False
                
        # Check right child
        if right_child < n:
            if arr[root_index] < arr[right_child]:
                return False
            if not self._is_subtree_max_heap(arr, right_child):
                return False
                
        return True

class TestHeapSortComparison(unittest.TestCase):
    """Test cases for heapsort comparison with other algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = HeapSortAnalyzer()
        
    def test_heapsort_vs_merge_sort(self):
        """Test that heapsort and merge sort produce same results"""
        test_arrays = [
            [3, 1, 4, 1, 5, 9, 2, 6, 5],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [7, 7, 7, 7],
            [-2, 0, 5, -1, 3],
            [random.randint(1, 50) for _ in range(30)]
        ]
        
        for arr in test_arrays:
            heap_result = self.analyzer.heapsort(arr.copy())
            merge_result = self.analyzer.merge_sort(arr.copy())
            expected = sorted(arr)
            
            self.assertEqual(heap_result, expected)
            self.assertEqual(merge_result, expected)
            self.assertEqual(heap_result, merge_result)
            
    def test_heapsort_vs_quicksort(self):
        """Test that heapsort and quicksort produce same results"""
        test_arrays = [
            [8, 3, 5, 4, 7, 6, 1, 2],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1],
            [random.randint(1, 100) for _ in range(25)]
        ]
        
        for arr in test_arrays:
            heap_result = self.analyzer.heapsort(arr.copy())
            quick_result = self.analyzer.quicksort(arr.copy())
            expected = sorted(arr)
            
            self.assertEqual(heap_result, expected)
            self.assertEqual(quick_result, expected)
            self.assertEqual(heap_result, quick_result)

class TestHeapSortPerformance(unittest.TestCase):
    """Test cases for heapsort performance characteristics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = HeapSortAnalyzer()
        
    def test_performance_consistency(self):
        """Test that heapsort performance is consistent across input types"""
        size = 100
        num_trials = 5
        
        # Generate different input types
        test_arrays = {
            'random': [random.randint(1, 1000) for _ in range(size)],
            'sorted': list(range(1, size + 1)),
            'reverse': list(range(size, 0, -1)),
            'duplicates': [random.randint(1, 10) for _ in range(size)]
        }
        
        times = {}
        
        for array_type, arr in test_arrays.items():
            trial_times = []
            
            for _ in range(num_trials):
                start_time = time.perf_counter()
                self.analyzer.heapsort(arr.copy())
                end_time = time.perf_counter()
                trial_times.append(end_time - start_time)
                
            times[array_type] = sum(trial_times) / len(trial_times)
        
        # Check that times are relatively consistent (within 3x of each other)
        min_time = min(times.values())
        max_time = max(times.values())
        
        # Heapsort should have relatively consistent performance
        self.assertLess(max_time / min_time, 3.0, 
                       f"Performance too inconsistent: {times}")
        
    def test_memory_usage(self):
        """Test that heapsort uses constant extra memory"""
        # This test verifies that heapsort modifies the input array in-place
        arr = [5, 2, 8, 1, 9, 3]
        original_id = id(arr)
        
        result = self.analyzer.heapsort(arr)
        
        # Check that same array object is returned (in-place sorting)
        self.assertEqual(id(result), original_id)
        
        # Check that array is actually sorted
        self.assertEqual(result, [1, 2, 3, 5, 8, 9])

class TestInputGeneration(unittest.TestCase):
    """Test cases for test array generation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = HeapSortAnalyzer()
        
    def test_generate_test_arrays(self):
        """Test test array generation functionality"""
        size = 10
        arrays = self.analyzer.generate_test_arrays(size)
        
        # Check all expected array types are generated
        expected_types = ['random', 'sorted', 'reverse_sorted', 'nearly_sorted', 'duplicates']
        for array_type in expected_types:
            self.assertIn(array_type, arrays)
            self.assertEqual(len(arrays[array_type]), size)
            
        # Verify sorted array is actually sorted
        self.assertEqual(arrays['sorted'], list(range(1, size + 1)))
        
        # Verify reverse sorted array
        self.assertEqual(arrays['reverse_sorted'], list(range(size, 0, -1)))
        
        # Verify duplicates array has repeated elements
        unique_elements = len(set(arrays['duplicates']))
        self.assertLess(unique_elements, size, "Duplicates array should have repeated elements")

if __name__ == '__main__':
    # Run all tests with detailed output
    unittest.main(verbosity=2)