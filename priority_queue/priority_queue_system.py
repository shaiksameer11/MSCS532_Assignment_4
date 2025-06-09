import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class Priority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Task:
    """
    Task class to represent individual tasks in priority queue
    """
    task_id: int
    priority: int  # Higher number = higher priority
    arrival_time: float
    deadline: float
    description: str
    execution_time: float = 1.0  # How long task takes to complete
    
    def __lt__(self, other):
        """Comparison method for heap operations (max-heap behavior)"""
        return self.priority < other.priority
    
    def __repr__(self):
        return f"Task({self.task_id}, P:{self.priority}, Deadline:{self.deadline:.1f})"

class PriorityQueue:
    """
    Priority Queue implementation using binary max-heap
    Uses array-based representation for efficiency
    """
    
    def __init__(self, use_max_heap: bool = True):
        """
        Initialize priority queue
        
        Parameters:
            use_max_heap: True for max-heap (highest priority first),
                         False for min-heap (lowest priority first)
        """
        self.heap = []  # Array to store heap elements
        self.use_max_heap = use_max_heap
        
        # Statistics for analysis
        self.total_operations = 0
        self.insert_operations = 0
        self.extract_operations = 0
        self.key_change_operations = 0
        
    def insert(self, task: Task):
        """
        Insert new task into priority queue
        
        Time Complexity: O(log n)
        
        Parameters:
            task: Task object to insert
        """
        self.heap.append(task)
        self.total_operations += 1
        self.insert_operations += 1
        
        # Restore heap property by moving element up
        self._heapify_up(len(self.heap) - 1)
    
    def extract_max(self) -> Optional[Task]:
        """
        Remove and return task with highest priority
        
        Time Complexity: O(log n)
        
        Returns:
            Task with highest priority, or None if queue is empty
        """
        if self.is_empty():
            return None
            
        self.total_operations += 1
        self.extract_operations += 1
        
        # Store the maximum element (root)
        max_task = self.heap[0]
        
        # Move last element to root
        last_element = self.heap.pop()
        
        if not self.is_empty():
            self.heap[0] = last_element
            # Restore heap property by moving element down
            self._heapify_down(0)
        
        return max_task
    
    def extract_min(self) -> Optional[Task]:
        """
        Remove and return task with lowest priority
        Only works correctly if use_max_heap=False
        
        Time Complexity: O(log n)
        
        Returns:
            Task with lowest priority, or None if queue is empty
        """
        if not self.use_max_heap:
            return self.extract_max()
        else:
            raise ValueError("extract_min() only works with min-heap configuration")
    
    def increase_key(self, task_index: int, new_priority: int):
        """
        Increase priority of task at given index (for max-heap)
        
        Time Complexity: O(log n)
        
        Parameters:
            task_index: Index of task in heap array
            new_priority: New priority value (must be higher than current)
        """
        if task_index >= len(self.heap):
            raise IndexError("Task index out of range")
            
        task = self.heap[task_index]
        
        if self.use_max_heap and new_priority < task.priority:
            raise ValueError("New priority must be higher than current priority for max-heap")
        
        # Update priority
        task.priority = new_priority
        self.total_operations += 1
        self.key_change_operations += 1
        
        # Restore heap property
        if self.use_max_heap:
            self._heapify_up(task_index)
        else:
            self._heapify_down(task_index)
    
    def decrease_key(self, task_index: int, new_priority: int):
        """
        Decrease priority of task at given index (for min-heap)
        
        Time Complexity: O(log n)
        
        Parameters:
            task_index: Index of task in heap array
            new_priority: New priority value (must be lower than current)
        """
        if task_index >= len(self.heap):
            raise IndexError("Task index out of range")
            
        task = self.heap[task_index]
        
        if not self.use_max_heap and new_priority > task.priority:
            raise ValueError("New priority must be lower than current priority for min-heap")
        
        # Update priority
        task.priority = new_priority
        self.total_operations += 1
        self.key_change_operations += 1
        
        # Restore heap property
        if self.use_max_heap:
            self._heapify_down(task_index)
        else:
            self._heapify_up(task_index)
    
    def peek(self) -> Optional[Task]:
        """
        Return highest priority task without removing it
        
        Time Complexity: O(1)
        
        Returns:
            Task with highest priority, or None if queue is empty
        """
        return self.heap[0] if not self.is_empty() else None
    
    def is_empty(self) -> bool:
        """
        Check if priority queue is empty
        
        Time Complexity: O(1)
        
        Returns:
            True if queue is empty, False otherwise
        """
        return len(self.heap) == 0
    
    def size(self) -> int:
        """Get number of tasks in queue"""
        return len(self.heap)
    
    def _heapify_up(self, index: int):
        """
        Move element up the heap to restore heap property
        Used after insertion
        
        Parameters:
            index: Index of element to move up
        """
        if index == 0:  # Already at root
            return
            
        parent_index = (index - 1) // 2
        
        # Check if heap property is violated
        if self._should_swap_up(index, parent_index):
            # Swap with parent
            self.heap[index], self.heap[parent_index] = \
                self.heap[parent_index], self.heap[index]
            
            # Continue heapifying up
            self._heapify_up(parent_index)
    
    def _heapify_down(self, index: int):
        """
        Move element down the heap to restore heap property
        Used after extraction
        
        Parameters:
            index: Index of element to move down
        """
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        target_index = index
        
        # Find the appropriate child to swap with
        if (left_child < len(self.heap) and 
            self._should_swap_down(left_child, target_index)):
            target_index = left_child
            
        if (right_child < len(self.heap) and 
            self._should_swap_down(right_child, target_index)):
            target_index = right_child
        
        # If we need to swap, do it and continue
        if target_index != index:
            self.heap[index], self.heap[target_index] = \
                self.heap[target_index], self.heap[index]
            
            self._heapify_down(target_index)
    
    def _should_swap_up(self, child_index: int, parent_index: int) -> bool:
        """Check if child should be swapped with parent"""
        child_priority = self.heap[child_index].priority
        parent_priority = self.heap[parent_index].priority
        
        if self.use_max_heap:
            return child_priority > parent_priority
        else:
            return child_priority < parent_priority
    
    def _should_swap_down(self, child_index: int, parent_index: int) -> bool:
        """Check if parent should be swapped with child"""
        child_priority = self.heap[child_index].priority
        parent_priority = self.heap[parent_index].priority
        
        if self.use_max_heap:
            return child_priority > parent_priority
        else:
            return child_priority < parent_priority
    
    def get_statistics(self) -> dict:
        """Get detailed statistics about priority queue operations"""
        return {
            'total_operations': self.total_operations,
            'insert_operations': self.insert_operations,
            'extract_operations': self.extract_operations,
            'key_change_operations': self.key_change_operations,
            'current_size': len(self.heap),
            'heap_type': 'max-heap' if self.use_max_heap else 'min-heap'
        }
    
    def display_heap(self):
        """Display current heap contents for debugging"""
        if self.is_empty():
            print("Priority queue is empty")
            return
            
        print(f"Priority Queue ({'Max-Heap' if self.use_max_heap else 'Min-Heap'}):")
        print("=" * 50)
        
        for i, task in enumerate(self.heap):
            level = int(np.log2(i + 1))
            indent = "  " * level
            print(f"{indent}[{i}] {task}")

class TaskScheduler:
    """
    Task scheduling system using priority queue
    Demonstrates real-world application of heap data structure
    """
    
    def __init__(self, scheduling_policy: str = "priority"):
        """
        Initialize task scheduler
        
        Parameters:
            scheduling_policy: "priority", "deadline", or "fifo"
        """
        self.scheduling_policy = scheduling_policy
        self.priority_queue = PriorityQueue(use_max_heap=True)
        self.completed_tasks = []
        self.current_time = 0.0
        self.total_waiting_time = 0.0
        self.total_turnaround_time = 0.0
        
    def add_task(self, task: Task):
        """
        Add new task to scheduler
        
        Parameters:
            task: Task to add to schedule
        """
        # Adjust priority based on scheduling policy
        if self.scheduling_policy == "deadline":
            # Earlier deadline = higher priority
            task.priority = int(1000 / max(task.deadline - self.current_time, 1))
        elif self.scheduling_policy == "fifo":
            # First come, first served
            task.priority = int(-task.arrival_time)  # Negative for FIFO behavior
            
        self.priority_queue.insert(task)
        print(f"Added {task} at time {self.current_time:.1f}")
    
    def execute_next_task(self) -> Optional[Task]:
        """
        Execute the next highest priority task
        
        Returns:
            Executed task, or None if no tasks available
        """
        if self.priority_queue.is_empty():
            return None
            
        task = self.priority_queue.extract_max()
        
        # Calculate performance metrics
        waiting_time = self.current_time - task.arrival_time
        self.current_time += task.execution_time
        turnaround_time = self.current_time - task.arrival_time
        
        self.total_waiting_time += waiting_time
        self.total_turnaround_time += turnaround_time
        
        print(f"Executed {task} at time {self.current_time:.1f} "
              f"(waited {waiting_time:.1f}s)")
        
        self.completed_tasks.append(task)
        return task
    
    def simulate_scheduling(self, tasks: List[Task], verbose: bool = True):
        """
        Simulate task scheduling process
        
        Parameters:
            tasks: List of tasks to schedule
            verbose: Whether to print detailed execution log
        """
        if verbose:
            print(f"\n=== TASK SCHEDULING SIMULATION ({self.scheduling_policy.upper()}) ===")
        
        # Add all tasks to scheduler
        for task in tasks:
            self.current_time = task.arrival_time
            self.add_task(task)
        
        if verbose:
            print(f"\nExecution Order:")
            print("-" * 30)
        
        # Execute all tasks
        while not self.priority_queue.is_empty():
            self.execute_next_task()
        
        # Calculate and display metrics
        num_tasks = len(self.completed_tasks)
        avg_waiting_time = self.total_waiting_time / num_tasks if num_tasks > 0 else 0
        avg_turnaround_time = self.total_turnaround_time / num_tasks if num_tasks > 0 else 0
        
        if verbose:
            print(f"\nScheduling Results:")
            print(f"  Average waiting time: {avg_waiting_time:.2f} seconds")
            print(f"  Average turnaround time: {avg_turnaround_time:.2f} seconds")
            print(f"  Total completion time: {self.current_time:.2f} seconds")
        
        return {
            'avg_waiting_time': avg_waiting_time,
            'avg_turnaround_time': avg_turnaround_time,
            'total_time': self.current_time,
            'completed_tasks': len(self.completed_tasks)
        }

class PriorityQueueAnalyzer:
    """
    Analyzer for priority queue performance and applications
    """
    
    def generate_random_tasks(self, num_tasks: int, max_priority: int = 10) -> List[Task]:
        """
        Generate random tasks for testing
        
        Parameters:
            num_tasks: Number of tasks to generate
            max_priority: Maximum priority value
            
        Returns:
            List of random tasks
        """
        tasks = []
        
        for i in range(num_tasks):
            task = Task(
                task_id=i,
                priority=random.randint(1, max_priority),
                arrival_time=random.uniform(0, 10),
                deadline=random.uniform(5, 20),
                description=f"Task {i}",
                execution_time=random.uniform(0.5, 3.0)
            )
            tasks.append(task)
        
        return sorted(tasks, key=lambda t: t.arrival_time)
    
    def benchmark_operations(self, sizes: List[int], num_trials: int = 5) -> dict:
        """
        Benchmark priority queue operations performance
        
        Parameters:
            sizes: List of queue sizes to test
            num_trials: Number of trials per size
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'sizes': sizes,
            'insert_times': [],
            'extract_times': [],
            'peek_times': []
        }
        
        for size in sizes:
            print(f"Benchmarking priority queue with {size} elements...")
            
            insert_times = []
            extract_times = []
            peek_times = []
            
            for trial in range(num_trials):
                pq = PriorityQueue()
                tasks = self.generate_random_tasks(size)
                
                # Benchmark insertions
                start_time = time.perf_counter()
                for task in tasks:
                    pq.insert(task)
                insert_time = time.perf_counter() - start_time
                insert_times.append(insert_time / size)  # Per operation
                
                # Benchmark peek operations
                start_time = time.perf_counter()
                for _ in range(min(1000, size)):
                    pq.peek()
                peek_time = time.perf_counter() - start_time
                peek_times.append(peek_time / min(1000, size))  # Per operation
                
                # Benchmark extractions
                start_time = time.perf_counter()
                while not pq.is_empty():
                    pq.extract_max()
                extract_time = time.perf_counter() - start_time
                extract_times.append(extract_time / size)  # Per operation
            
            results['insert_times'].append(np.mean(insert_times))
            results['extract_times'].append(np.mean(extract_times))
            results['peek_times'].append(np.mean(peek_times))
        
        return results
    
    def compare_scheduling_policies(self, num_tasks: int = 20) -> dict:
        """
        Compare different scheduling policies
        
        Parameters:
            num_tasks: Number of tasks to test with
            
        Returns:
            Dictionary with comparison results
        """
        tasks = self.generate_random_tasks(num_tasks)
        policies = ["priority", "deadline", "fifo"]
        results = {}
        
        print("=== SCHEDULING POLICY COMPARISON ===\n")
        
        for policy in policies:
            scheduler = TaskScheduler(scheduling_policy=policy)
            # Make copies of tasks since they get modified during scheduling
            task_copies = [Task(t.task_id, t.priority, t.arrival_time, 
                              t.deadline, t.description, t.execution_time) 
                          for t in tasks]
            
            metrics = scheduler.simulate_scheduling(task_copies, verbose=False)
            results[policy] = metrics
            
            print(f"{policy.upper()} Scheduling:")
            print(f"  Average waiting time: {metrics['avg_waiting_time']:.2f}s")
            print(f"  Average turnaround time: {metrics['avg_turnaround_time']:.2f}s")
            print(f"  Total completion time: {metrics['total_time']:.2f}s")
            print()
        
        return results
    
    def analyze_complexity(self):
        """
        Analyze time complexity of priority queue operations
        """
        print("=== PRIORITY QUEUE TIME COMPLEXITY ANALYSIS ===\n")
        
        print("Data Structure Choice: Array-based Binary Heap")
        print("=============================================")
        print("Why Array over Linked List:")
        print("- Better cache performance due to memory locality")
        print("- Simple index calculations for parent/child relationships")
        print("- No extra memory overhead for pointers")
        print("- Easy to implement and understand")
        print()
        
        print("Parent-Child Relationships in Array:")
        print("====================================")
        print("For element at index i:")
        print("- Parent index: (i - 1) // 2")
        print("- Left child index: 2 * i + 1")
        print("- Right child index: 2 * i + 2")
        print()
        
        print("Operation Time Complexities:")
        print("===========================")
        
        print("1. INSERT Operation: O(log n)")
        print("   - Add element at end of array: O(1)")
        print("   - Bubble up to restore heap property: O(log n)")
        print("   - Maximum height of heap: log n")
        print()
        
        print("2. EXTRACT_MAX Operation: O(log n)")
        print("   - Remove root element: O(1)")
        print("   - Move last element to root: O(1)")
        print("   - Bubble down to restore heap property: O(log n)")
        print("   - Maximum height traversed: log n")
        print()
        
        print("3. PEEK Operation: O(1)")
        print("   - Simply return root element")
        print("   - No modification needed")
        print()
        
        print("4. INCREASE_KEY/DECREASE_KEY: O(log n)")
        print("   - Update priority value: O(1)")
        print("   - Restore heap property: O(log n)")
        print("   - May need to traverse up or down the heap")
        print()
        
        print("5. IS_EMPTY: O(1)")
        print("   - Check array length")
        print()
        
        print("Space Complexity: O(n)")
        print("====================")
        print("- Store n elements in array")
        print("- No additional space needed for heap structure")
        print("- Very memory efficient")
        print()
        
        # Empirical validation
        print("Empirical Validation:")
        print("====================")
        sizes = [100, 500, 1000, 2000, 5000]
        
        for size in sizes:
            pq = PriorityQueue()
            tasks = self.generate_random_tasks(size)
            
            # Measure insert operations
            start_time = time.perf_counter()
            for task in tasks:
                pq.insert(task)
            insert_time = time.perf_counter() - start_time
            
            # Theoretical time for insert operations
            theoretical_time = size * np.log2(size) * 1e-6  # Approximate scaling
            
            print(f"Size {size}: Insert time = {insert_time:.6f}s, "
                  f"Theory approximately {theoretical_time:.6f}s, "
                  f"Ratio = {insert_time/theoretical_time:.2f}")
    
    def plot_performance_results(self, results: dict):
        """Create graphs showing priority queue performance"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        sizes = results['sizes']
        
        # Plot operation times
        ax1.plot(sizes, results['insert_times'], 'o-', label='Insert', linewidth=2, color='blue')
        ax1.plot(sizes, results['extract_times'], 's-', label='Extract Max', linewidth=2, color='red')
        ax1.plot(sizes, results['peek_times'], '^-', label='Peek', linewidth=2, color='green')
        ax1.set_xlabel('Queue Size')
        ax1.set_ylabel('Time per Operation (seconds)')
        ax1.set_title('Priority Queue Operation Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot theoretical vs empirical complexity
        theoretical_insert = [size * np.log2(size) * 1e-8 for size in sizes]
        ax2.plot(sizes, results['insert_times'], 'o-', label='Measured Insert Time', linewidth=2)
        ax2.plot(sizes, theoretical_insert, '--', label='O(log n) Theoretical', linewidth=2)
        ax2.set_xlabel('Queue Size')
        ax2.set_ylabel('Time per Operation (seconds)')
        ax2.set_title('Insert Time: Measured vs Theoretical')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Plot heap height vs size
        heap_heights = [np.log2(size) for size in sizes]
        ax3.plot(sizes, heap_heights, 'o-', linewidth=2, color='purple')
        ax3.set_xlabel('Queue Size')
        ax3.set_ylabel('Heap Height')
        ax3.set_title('Heap Height vs Size')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot memory usage
        memory_usage = [size * 8 / 1024 for size in sizes]  # Assuming 8 bytes per element
        ax4.plot(sizes, memory_usage, 'o-', linewidth=2, color='orange')
        ax4.set_xlabel('Queue Size')
        ax4.set_ylabel('Memory Usage (KB)')
        ax4.set_title('Memory Usage vs Size')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('results/priority_queue_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_applications(self):
        """Demonstrate real-world applications of priority queues"""
        print("=== REAL-WORLD APPLICATIONS OF PRIORITY QUEUES ===\n")
        
        print("1. Task Scheduling in Operating Systems")
        print("======================================")
        print("- CPU scheduling based on process priority")
        print("- Higher priority processes get CPU time first")
        print("- Real-time systems use priority queues for deadline management")
        print()
        
        print("2. Dijkstra's Shortest Path Algorithm")
        print("====================================")
        print("- Find shortest path in weighted graphs")
        print("- Extract minimum distance vertex efficiently")
        print("- Used in GPS navigation and network routing")
        print()
        
        print("3. A* Pathfinding Algorithm")
        print("==========================")
        print("- Game AI and robotics pathfinding")
        print("- Priority based on estimated total cost")
        print("- Efficiently explores most promising paths first")
        print()
        
        print("4. Huffman Coding for Data Compression")
        print("=====================================")
        print("- Build optimal binary trees for compression")
        print("- Merge nodes with lowest frequency first")
        print("- Used in ZIP files and JPEG compression")
        print()
        
        print("5. Event-Driven Simulation")
        print("==========================")
        print("- Process events in chronological order")
        print("- Network simulation and discrete event systems")
        print("- Efficiently manage future event scheduling")
        print()
        
        print("6. Emergency Room Patient Management")
        print("===================================")
        print("- Triage patients based on severity")
        print("- Life-threatening cases get immediate attention")
        print("- Optimize patient flow and resource allocation")
        print()

# Example usage and testing
if __name__ == "__main__":
    print("=== PRIORITY QUEUE BASIC OPERATIONS TEST ===")
    
    # Create priority queue and test basic operations
    pq = PriorityQueue()
    
    # Create sample tasks
    tasks = [
        Task(1, 3, 0.0, 10.0, "Medium priority task"),
        Task(2, 5, 1.0, 8.0, "High priority task"),
        Task(3, 1, 2.0, 15.0, "Low priority task"),
        Task(4, 4, 3.0, 12.0, "Important task"),
        Task(5, 2, 4.0, 20.0, "Regular task")
    ]
    
    # Test insertions
    print("Testing insertions...")
    for task in tasks:
        pq.insert(task)
        print(f"Inserted: {task}")
    
    print(f"\nPriority queue size: {pq.size()}")
    print(f"Highest priority task: {pq.peek()}")
    
    # Display heap structure
    print("\nHeap structure:")
    pq.display_heap()
    
    # Test extractions
    print("\nTesting extractions...")
    while not pq.is_empty():
        task = pq.extract_max()
        print(f"Extracted: {task}")
    
    print("\n" + "="*60)
    
    # Run theoretical analysis
    analyzer = PriorityQueueAnalyzer()
    analyzer.analyze_complexity()
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    # Benchmark operations
    sizes = [100, 500, 1000, 2000, 5000]
    results = analyzer.benchmark_operations(sizes, num_trials=3)
    
    # Plot performance results
    analyzer.plot_performance_results(results)
    
    # Compare scheduling policies
    print("\n" + "="*60)
    print("SCHEDULING POLICY COMPARISON")
    print("="*60)
    
    scheduler_results = analyzer.compare_scheduling_policies(20)
    
    # Demonstrate applications
    print("\n" + "="*60)
    analyzer.demonstrate_applications()
    
    # Show statistics
    print("\nFinal Performance Summary:")
    print("=========================")
    for i, size in enumerate(sizes):
        print(f"Size {size}:")
        print(f"  Insert time: {results['insert_times'][i]:.8f}s per operation")
        print(f"  Extract time: {results['extract_times'][i]:.8f}s per operation")
        print(f"  Peek time: {results['peek_times'][i]:.8f}s per operation")
        print()