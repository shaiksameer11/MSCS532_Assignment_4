"""
Unit tests for Priority Queue implementation
Tests correctness, edge cases, and performance characteristics
"""

import unittest
import sys
import os
import random
import time

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'priority_queue'))

from priority_queue_system import PriorityQueue, Task, TaskScheduler, Priority

class TestTask(unittest.TestCase):
    """Test cases for Task class"""
    
    def test_task_creation(self):
        """Test task creation with all parameters"""
        task = Task(
            task_id=1,
            priority=5,
            arrival_time=0.0,
            deadline=10.0,
            description="Test task",
            execution_time=2.0
        )
        
        self.assertEqual(task.task_id, 1)
        self.assertEqual(task.priority, 5)
        self.assertEqual(task.arrival_time, 0.0)
        self.assertEqual(task.deadline, 10.0)
        self.assertEqual(task.description, "Test task")
        self.assertEqual(task.execution_time, 2.0)
        
    def test_task_comparison(self):
        """Test task comparison for heap operations"""
        task1 = Task(1, 5, 0.0, 10.0, "High priority")
        task2 = Task(2, 3, 1.0, 12.0, "Low priority")
        
        # task1 has higher priority, so should be "greater" for max-heap
        self.assertFalse(task1 < task2)  # For max-heap behavior
        self.assertTrue(task2 < task1)
        
    def test_task_string_representation(self):
        """Test task string representation"""
        task = Task(1, 5, 0.0, 10.0, "Test task")
        str_repr = str(task)
        
        self.assertIn("Task(1", str_repr)
        self.assertIn("P:5", str_repr)
        self.assertIn("Deadline:10.0", str_repr)

class TestPriorityQueueBasic(unittest.TestCase):
    """Test cases for basic priority queue operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pq = PriorityQueue()
        
    def test_empty_queue(self):
        """Test operations on empty priority queue"""
        self.assertTrue(self.pq.is_empty())
        self.assertEqual(self.pq.size(), 0)
        self.assertIsNone(self.pq.peek())
        self.assertIsNone(self.pq.extract_max())
        
    def test_single_element(self):
        """Test operations with single element"""
        task = Task(1, 5, 0.0, 10.0, "Single task")
        
        self.pq.insert(task)
        
        self.assertFalse(self.pq.is_empty())
        self.assertEqual(self.pq.size(), 1)
        self.assertEqual(self.pq.peek(), task)
        
        extracted = self.pq.extract_max()
        self.assertEqual(extracted, task)
        self.assertTrue(self.pq.is_empty())
        
    def test_multiple_insertions(self):
        """Test multiple insertions and priority ordering"""
        tasks = [
            Task(1, 3, 0.0, 10.0, "Medium priority"),
            Task(2, 5, 1.0, 8.0, "High priority"),
            Task(3, 1, 2.0, 15.0, "Low priority"),
            Task(4, 4, 3.0, 12.0, "Higher medium priority")
        ]
        
        # Insert all tasks
        for task in tasks:
            self.pq.insert(task)
            
        self.assertEqual(self.pq.size(), 4)
        
        # Extract tasks - should come out in priority order
        extracted_priorities = []
        while not self.pq.is_empty():
            task = self.pq.extract_max()
            extracted_priorities.append(task.priority)
            
        # Should be in descending priority order
        self.assertEqual(extracted_priorities, [5, 4, 3, 1])
        
    def test_same_priority_tasks(self):
        """Test handling of tasks with same priority"""
        task1 = Task(1, 5, 0.0, 10.0, "Task 1")
        task2 = Task(2, 5, 1.0, 11.0, "Task 2")
        task3 = Task(3, 5, 2.0, 9.0, "Task 3")
        
        self.pq.insert(task1)
        self.pq.insert(task2)
        self.pq.insert(task3)
        
        # All tasks should be extractable
        extracted_count = 0
        while not self.pq.is_empty():
            task = self.pq.extract_max()
            self.assertEqual(task.priority, 5)
            extracted_count += 1
            
        self.assertEqual(extracted_count, 3)
        
    def test_peek_operation(self):
        """Test peek operation doesn't modify queue"""
        tasks = [
            Task(1, 3, 0.0, 10.0, "Medium"),
            Task(2, 5, 1.0, 8.0, "High"),
            Task(3, 1, 2.0, 15.0, "Low")
        ]
        
        for task in tasks:
            self.pq.insert(task)
            
        initial_size = self.pq.size()
        
        # Peek multiple times
        for _ in range(3):
            peeked = self.pq.peek()
            self.assertEqual(peeked.priority, 5)  # Should always be highest priority
            self.assertEqual(self.pq.size(), initial_size)  # Size shouldn't change

class TestPriorityQueueAdvanced(unittest.TestCase):
    """Test cases for advanced priority queue operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pq = PriorityQueue()
        
    def test_increase_key_operation(self):
        """Test increase key operation"""
        tasks = [
            Task(1, 3, 0.0, 10.0, "Task 1"),
            Task(2, 5, 1.0, 8.0, "Task 2"),
            Task(3, 1, 2.0, 15.0, "Task 3")
        ]
        
        for task in tasks:
            self.pq.insert(task)
            
        # Increase priority of task at index 2 (lowest priority task)
        self.pq.increase_key(2, 10)
        
        # Now the task with priority 10 should be at the top
        highest = self.pq.peek()
        self.assertEqual(highest.priority, 10)
        
    def test_increase_key_invalid_index(self):
        """Test increase key with invalid index"""
        task = Task(1, 5, 0.0, 10.0, "Task")
        self.pq.insert(task)
        
        with self.assertRaises(IndexError):
            self.pq.increase_key(5, 10)  # Index out of range
            
    def test_increase_key_lower_priority(self):
        """Test increase key with lower priority (should raise error)"""
        task = Task(1, 5, 0.0, 10.0, "Task")
        self.pq.insert(task)
        
        with self.assertRaises(ValueError):
            self.pq.increase_key(0, 3)  # Can't decrease priority in max-heap
            
    def test_min_heap_configuration(self):
        """Test priority queue configured as min-heap"""
        min_pq = PriorityQueue(use_max_heap=False)
        
        tasks = [
            Task(1, 3, 0.0, 10.0, "Medium"),
            Task(2, 5, 1.0, 8.0, "High"),
            Task(3, 1, 2.0, 15.0, "Low")
        ]
        
        for task in tasks:
            min_pq.insert(task)
            
        # In min-heap, lowest priority should come out first
        extracted = min_pq.extract_max()  # Still called extract_max but behaves as min
        self.assertEqual(extracted.priority, 1)
        
    def test_large_number_of_elements(self):
        """Test priority queue with large number of elements"""
        num_elements = 1000
        tasks = []
        
        # Create tasks with random priorities
        for i in range(num_elements):
            task = Task(i, random.randint(1, 100), 0.0, 10.0, f"Task {i}")
            tasks.append(task)
            self.pq.insert(task)
            
        self.assertEqual(self.pq.size(), num_elements)
        
        # Extract all elements and verify they come out in priority order
        extracted_priorities = []
        while not self.pq.is_empty():
            task = self.pq.extract_max()
            extracted_priorities.append(task.priority)
            
        # Check that priorities are in non-increasing order
        for i in range(1, len(extracted_priorities)):
            self.assertGreaterEqual(extracted_priorities[i-1], extracted_priorities[i])

class TestPriorityQueuePerformance(unittest.TestCase):
    """Test cases for priority queue performance characteristics"""
    
    def test_insert_performance(self):
        """Test insert operation performance scaling"""
        pq = PriorityQueue()
        sizes = [100, 200, 400, 800]
        times = []
        
        for size in sizes:
            tasks = [Task(i, random.randint(1, 100), 0.0, 10.0, f"Task {i}") 
                    for i in range(size)]
            
            start_time = time.perf_counter()
            for task in tasks:
                pq.insert(task)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
        # Insert time should scale roughly as O(n log n) for n insertions
        # Check that time doesn't grow faster than O(n^2)
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Time ratio should be less than size_ratio^2 (quadratic growth)
            self.assertLess(time_ratio, size_ratio**2,
                           f"Insert performance worse than O(n^2): {time_ratio} vs {size_ratio**2}")
            
    def test_extract_performance(self):
        """Test extract operation performance"""
        pq = PriorityQueue()
        size = 1000
        
        # Fill queue
        for i in range(size):
            task = Task(i, random.randint(1, 100), 0.0, 10.0, f"Task {i}")
            pq.insert(task)
            
        # Measure extraction time
        start_time = time.perf_counter()
        while not pq.is_empty():
            pq.extract_max()
        end_time = time.perf_counter()
        
        extraction_time = end_time - start_time
        
        # Should complete within reasonable time (less than 1 second for 1000 elements)
        self.assertLess(extraction_time, 1.0,
                       f"Extraction too slow: {extraction_time:.3f} seconds for {size} elements")

class TestTaskScheduler(unittest.TestCase):
    """Test cases for TaskScheduler class"""
    
    def test_priority_based_scheduling(self):
        """Test priority-based task scheduling"""
        scheduler = TaskScheduler("priority")
        
        tasks = [
            Task(1, 3, 0.0, 10.0, "Medium task", 1.0),
            Task(2, 5, 1.0, 8.0, "High task", 1.0),
            Task(3, 1, 2.0, 15.0, "Low task", 1.0),
            Task(4, 4, 3.0, 12.0, "High-medium task", 1.0)
        ]
        
        results = scheduler.simulate_scheduling(tasks, verbose=False)
        
        # Check that scheduling completed
        self.assertEqual(results['completed_tasks'], 4)
        self.assertGreater(results['total_time'], 0)
        self.assertGreater(results['avg_waiting_time'], 0)
        self.assertGreater(results['avg_turnaround_time'], 0)
        
    def test_deadline_based_scheduling(self):
        """Test deadline-based task scheduling"""
        scheduler = TaskScheduler("deadline")
        
        tasks = [
            Task(1, 3, 0.0, 15.0, "Late deadline", 2.0),
            Task(2, 5, 1.0, 5.0, "Early deadline", 1.0),
            Task(3, 1, 2.0, 8.0, "Medium deadline", 1.5)
        ]
        
        results = scheduler.simulate_scheduling(tasks, verbose=False)
        
        # All tasks should be completed
        self.assertEqual(results['completed_tasks'], 3)
        
    def test_fifo_scheduling(self):
        """Test FIFO (first-in-first-out) scheduling"""
        scheduler = TaskScheduler("fifo")
        
        tasks = [
            Task(1, 5, 0.0, 10.0, "First arrival", 1.0),
            Task(2, 10, 2.0, 8.0, "Second arrival", 1.0),
            Task(3, 1, 4.0, 15.0, "Third arrival", 1.0)
        ]
        
        results = scheduler.simulate_scheduling(tasks, verbose=False)
        
        # All tasks should be completed
        self.assertEqual(results['completed_tasks'], 3)
        
    def test_empty_task_list(self):
        """Test scheduler with empty task list"""
        scheduler = TaskScheduler("priority")
        
        results = scheduler.simulate_scheduling([], verbose=False)
        
        self.assertEqual(results['completed_tasks'], 0)
        self.assertEqual(results['total_time'], 0.0)
        self.assertEqual(results['avg_waiting_time'], 0)
        self.assertEqual(results['avg_turnaround_time'], 0)
        
    def test_single_task_scheduling(self):
        """Test scheduler with single task"""
        scheduler = TaskScheduler("priority")
        
        task = Task(1, 5, 0.0, 10.0, "Single task", 2.0)
        results = scheduler.simulate_scheduling([task], verbose=False)
        
        self.assertEqual(results['completed_tasks'], 1)
        self.assertEqual(results['total_time'], 2.0)  # Should equal execution time
        self.assertEqual(results['avg_waiting_time'], 0.0)  # No waiting for single task
        self.assertEqual(results['avg_turnaround_time'], 2.0)  # Should equal execution time

class TestPriorityQueueStatistics(unittest.TestCase):
    """Test cases for priority queue statistics and monitoring"""
    
    def test_statistics_collection(self):
        """Test statistics collection functionality"""
        pq = PriorityQueue()
        
        # Perform various operations
        for i in range(5):
            task = Task(i, random.randint(1, 10), 0.0, 10.0, f"Task {i}")
            pq.insert(task)
            
        # Extract a few tasks
        pq.extract_max()
        pq.extract_max()
        
        # Modify priority
        if pq.size() > 0:
            pq.increase_key(0, 15)
        
        stats = pq.get_statistics()
        
        # Check that all expected statistics are present
        expected_keys = [
            'total_operations', 'insert_operations', 'extract_operations',
            'key_change_operations', 'current_size', 'heap_type'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            
        # Verify some basic relationships
        self.assertEqual(stats['insert_operations'], 5)
        self.assertEqual(stats['extract_operations'], 2)
        self.assertEqual(stats['key_change_operations'], 1)
        self.assertEqual(stats['current_size'], 3)
        self.assertEqual(stats['heap_type'], 'max-heap')
        
    def test_operation_counting(self):
        """Test operation counting accuracy"""
        pq = PriorityQueue()
        
        initial_stats = pq.get_statistics()
        self.assertEqual(initial_stats['total_operations'], 0)
        
        # Perform 10 insert operations
        for i in range(10):
            task = Task(i, i, 0.0, 10.0, f"Task {i}")
            pq.insert(task)
            
        stats_after_insert = pq.get_statistics()
        self.assertEqual(stats_after_insert['insert_operations'], 10)
        self.assertEqual(stats_after_insert['total_operations'], 10)
        
        # Perform 5 extract operations
        for _ in range(5):
            pq.extract_max()
            
        final_stats = pq.get_statistics()
        self.assertEqual(final_stats['extract_operations'], 5)
        self.assertEqual(final_stats['total_operations'], 15)

class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error conditions"""
    
    def test_extract_from_empty_queue(self):
        """Test extracting from empty queue"""
        pq = PriorityQueue()
        
        result = pq.extract_max()
        self.assertIsNone(result)
        
    def test_peek_empty_queue(self):
        """Test peeking at empty queue"""
        pq = PriorityQueue()
        
        result = pq.peek()
        self.assertIsNone(result)
        
    def test_very_large_priorities(self):
        """Test with very large priority values"""
        pq = PriorityQueue()
        
        large_priority = 10**9
        task = Task(1, large_priority, 0.0, 10.0, "Large priority task")
        
        pq.insert(task)
        extracted = pq.extract_max()
        
        self.assertEqual(extracted.priority, large_priority)
        
    def test_negative_priorities(self):
        """Test with negative priority values"""
        pq = PriorityQueue()
        
        tasks = [
            Task(1, -5, 0.0, 10.0, "Negative priority 1"),
            Task(2, -1, 1.0, 8.0, "Negative priority 2"),
            Task(3, -10, 2.0, 15.0, "Negative priority 3")
        ]
        
        for task in tasks:
            pq.insert(task)
            
        # Should extract in order: -1, -5, -10 (highest to lowest)
        extracted_priorities = []
        while not pq.is_empty():
            task = pq.extract_max()
            extracted_priorities.append(task.priority)
            
        self.assertEqual(extracted_priorities, [-1, -5, -10])
        
    def test_zero_priority(self):
        """Test with zero priority value"""
        pq = PriorityQueue()
        
        tasks = [
            Task(1, 0, 0.0, 10.0, "Zero priority"),
            Task(2, 1, 1.0, 8.0, "Positive priority"),
            Task(3, -1, 2.0, 15.0, "Negative priority")
        ]
        
        for task in tasks:
            pq.insert(task)
            
        # Should extract in order: 1, 0, -1
        extracted_priorities = []
        while not pq.is_empty():
            task = pq.extract_max()
            extracted_priorities.append(task.priority)
            
        self.assertEqual(extracted_priorities, [1, 0, -1])

class TestHeapPropertyMaintenance(unittest.TestCase):
    """Test cases to verify heap property is maintained"""
    
    def test_heap_property_after_insertions(self):
        """Test heap property is maintained after multiple insertions"""
        pq = PriorityQueue()
        
        # Insert random priorities
        priorities = [3, 7, 1, 9, 2, 8, 4, 6, 5]
        for i, priority in enumerate(priorities):
            task = Task(i, priority, 0.0, 10.0, f"Task {i}")
            pq.insert(task)
            
            # Verify heap property after each insertion
            self.assertTrue(self._is_max_heap(pq.heap))
            
    def test_heap_property_after_extractions(self):
        """Test heap property is maintained after extractions"""
        pq = PriorityQueue()
        
        # Insert tasks
        for i in range(10):
            task = Task(i, random.randint(1, 20), 0.0, 10.0, f"Task {i}")
            pq.insert(task)
            
        # Extract tasks one by one and verify heap property
        while pq.size() > 1:  # Keep at least one element
            pq.extract_max()
            self.assertTrue(self._is_max_heap(pq.heap))
            
    def test_heap_property_after_key_increase(self):
        """Test heap property after increasing key"""
        pq = PriorityQueue()
        
        # Insert several tasks
        priorities = [5, 3, 8, 1, 9, 2]
        for i, priority in enumerate(priorities):
            task = Task(i, priority, 0.0, 10.0, f"Task {i}")
            pq.insert(task)
            
        # Increase priority of a task
        pq.increase_key(3, 15)  # Increase priority of task at index 3
        
        # Verify heap property is maintained
        self.assertTrue(self._is_max_heap(pq.heap))
        
    def _is_max_heap(self, heap):
        """Helper method to verify max heap property"""
        n = len(heap)
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            # Check left child
            if left_child < n and heap[i].priority < heap[left_child].priority:
                return False
                
            # Check right child
            if right_child < n and heap[i].priority < heap[right_child].priority:
                return False
                
        return True

class TestTaskSchedulerPolicies(unittest.TestCase):
    """Test cases for different scheduling policies"""
    
    def test_priority_policy_ordering(self):
        """Test that priority policy executes high priority tasks first"""
        scheduler = TaskScheduler("priority")
        
        tasks = [
            Task(1, 1, 0.0, 20.0, "Low priority", 1.0),
            Task(2, 5, 0.0, 10.0, "High priority", 1.0),
            Task(3, 3, 0.0, 15.0, "Medium priority", 1.0)
        ]
        
        # Track execution order by overriding execute_next_task
        executed_priorities = []
        original_execute = scheduler.execute_next_task
        
        def track_execution():
            task = original_execute()
            if task:
                executed_priorities.append(task.priority)
            return task
            
        scheduler.execute_next_task = track_execution
        
        scheduler.simulate_scheduling(tasks, verbose=False)
        
        # Should execute in priority order: 5, 3, 1
        self.assertEqual(executed_priorities, [5, 3, 1])
        
    def test_deadline_policy_behavior(self):
        """Test deadline policy prioritizes earlier deadlines"""
        scheduler = TaskScheduler("deadline")
        
        tasks = [
            Task(1, 1, 0.0, 15.0, "Late deadline", 1.0),
            Task(2, 5, 0.0, 5.0, "Early deadline", 1.0),  # Should go first despite lower base priority
            Task(3, 3, 0.0, 10.0, "Medium deadline", 1.0)
        ]
        
        # Add tasks and check their adjusted priorities
        for task in tasks:
            original_priority = task.priority
            scheduler.add_task(task)
            # Deadline policy should have modified the priority
            
        # Task with earliest deadline should have highest adjusted priority
        highest_priority_task = scheduler.priority_queue.peek()
        self.assertEqual(highest_priority_task.deadline, 5.0)  # Earliest deadline

if __name__ == '__main__':
    # Run all tests with detailed output
    unittest.main(verbosity=2)