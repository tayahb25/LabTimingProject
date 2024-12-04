import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import time
import random
import statistics
import matplotlib.pyplot as plt
import csv

# -----------------------------------
# Utility Functions
# -----------------------------------
def nanosec_to_sec(ns):
    """Convert nanoseconds to seconds."""
    BILLION = 1_000_000_000
    return ns / BILLION

def generate_random_ints(size):
    """Generate a list of random integers."""
    return [random.randint(0, 1000) for _ in range(size)]

def plot_results(data, title, xlabel, ylabel, filename):
    """Save the timing results as a plot."""
    sizes, times = zip(*data)
    plt.figure()
    plt.plot(sizes, times, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the figure to release memory


def save_results_to_csv(data, filename):
    """Save the timing results to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "Time (seconds)"])
        writer.writerows(data)

# -----------------------------------
# Data Structure Implementations
# -----------------------------------

# Stack
class Stack:
    def __init__(self):
        self.stack = []
    
    def push(self, value):
        self.stack.append(value)
    
    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        raise IndexError("Pop from empty stack")
    
    def is_empty(self):
        return len(self.stack) == 0

# Queue
class Queue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, value):
        self.queue.append(value)
    
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        raise IndexError("Dequeue from empty queue")
    
    def is_empty(self):
        return len(self.queue) == 0

# Linked List
class LinkedListNode:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        if not self.head:
            self.head = LinkedListNode(value)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = LinkedListNode(value)

    def get_entry(self, index):
        current = self.head
        count = 0
        while current:
            if count == index:
                return current.value
            current = current.next
            count += 1
        raise IndexError("Index out of bounds")

# Max Heap
class MaxHeap:
    def __init__(self):
        self.heap = []

    def add(self, value):
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    def _heapify_up(self, index):
        parent = (index - 1) // 2
        while index > 0 and self.heap[index] > self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent

# Binary Search Tree
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = BSTNode(value)
        else:
            self._insert_rec(self.root, value)

    def _insert_rec(self, node, value):
        if value < node.value:
            if node.left:
                self._insert_rec(node.left, value)
            else:
                node.left = BSTNode(value)
        else:
            if node.right:
                self._insert_rec(node.right, value)
            else:
                node.right = BSTNode(value)

    def search(self, value):
        return self._search_rec(self.root, value)

    def _search_rec(self, node, value):
        if not node or node.value == value:
            return node
        if value < node.value:
            return self._search_rec(node.left, value)
        return self._search_rec(node.right, value)  # Added 'value' argument


# -----------------------------------
# Timing Functions (All Operations)
# -----------------------------------

def time_stack_pop():
    results = []
    for size in range(1000, 100001, 1000):
        stack = Stack()
        for val in generate_random_ints(size):
            stack.push(val)
        start_time = time.perf_counter_ns()
        stack.pop()
        end_time = time.perf_counter_ns()
        elapsed_time = nanosec_to_sec(end_time - start_time)
        results.append((size, elapsed_time))
    return results

def time_stack_pop_all():
    results = []
    for size in range(1000, 100001, 1000):
        stack = Stack()
        for val in generate_random_ints(size):
            stack.push(val)
        start_time = time.perf_counter_ns()
        while not stack.is_empty():
            stack.pop()
        end_time = time.perf_counter_ns()
        elapsed_time = nanosec_to_sec(end_time - start_time)
        results.append((size, elapsed_time))
    return results

def time_queue_enqueue():
    results = []
    for size in range(1000, 100001, 1000):
        queue = Queue()
        for val in generate_random_ints(size):
            queue.enqueue(val)
        start_time = time.perf_counter_ns()
        queue.enqueue(0)
        end_time = time.perf_counter_ns()
        elapsed_time = nanosec_to_sec(end_time - start_time)
        results.append((size, elapsed_time))
    return results

def time_linked_list_get_last():
    """
    Times the get_entry operation at the last index of a LinkedList.
    """
    results = []
    for size in range(1000, 20001, 2000):  # Max size is 20,000 with step 2000
        ll = LinkedList()
        for val in generate_random_ints(size):
            ll.append(val)

        # Warm-up phase
        for _ in range(5):
            ll.get_entry(size - 1)

        # Time get_entry at last index
        start_time = time.perf_counter_ns()
        ll.get_entry(size - 1)  # Access the last element
        end_time = time.perf_counter_ns()

        elapsed_time = nanosec_to_sec(end_time - start_time)
        results.append((size, elapsed_time))
    return results



def time_linked_list_print_all():
    """
    Times the operation of traversing and printing all elements in a LinkedList.
    """
    results = []
    for size in range(1000, 20001, 2000):  # Max size 20,000
        ll = LinkedList()
        for val in generate_random_ints(size):
            ll.append(val)

        # Warm-up phase
        current = ll.head
        while current:
            current = current.next

        # Time printing all elements
        start_time = time.perf_counter_ns()
        current = ll.head
        while current:
            current = current.next
        end_time = time.perf_counter_ns()

        elapsed_time = nanosec_to_sec(end_time - start_time)
        results.append((size, elapsed_time))
    return results


def time_max_heap_add():
    results = []
    for size in range(1000, 100001, 1000):
        heap = MaxHeap()
        for val in generate_random_ints(size):
            heap.add(val)
        start_time = time.perf_counter_ns()
        heap.add(random.randint(0, 1000))
        end_time = time.perf_counter_ns()
        elapsed_time = nanosec_to_sec(end_time - start_time)
        results.append((size, elapsed_time))
    return results

def time_bst_search():
    results = []
    for size in range(1000, 100001, 1000):
        bst = BinarySearchTree()
        data = generate_random_ints(size)
        for val in data:
            bst.insert(val)
        target = random.choice(data)
        start_time = time.perf_counter_ns()
        bst.search(target)
        end_time = time.perf_counter_ns()
        elapsed_time = nanosec_to_sec(end_time - start_time)
        results.append((size, elapsed_time))
    return results

# -----------------------------------
# Main Execution
# -----------------------------------

if __name__ == "__main__":
    print("Starting timing for Stack Pop operation...")
    stack_pop_times = time_stack_pop()
    plot_results(stack_pop_times, "Stack Pop Timing", "Size of Stack", "Time (seconds)", "stack_pop_timing.png")
    save_results_to_csv(stack_pop_times, "stack_pop_timing.csv")

    print("Starting timing for Stack Pop All operation...")
    stack_pop_all_times = time_stack_pop_all()
    plot_results(stack_pop_all_times, "Stack Pop All Timing", "Size of Stack", "Time (seconds)", "stack_pop_all_timing.png")
    save_results_to_csv(stack_pop_all_times, "stack_pop_all_timing.csv")

    print("Starting timing for Queue Enqueue operation...")
    queue_enqueue_times = time_queue_enqueue()
    plot_results(queue_enqueue_times, "Queue Enqueue Timing", "Size of Queue", "Time (seconds)", "queue_enqueue_timing.png")
    save_results_to_csv(queue_enqueue_times, "queue_enqueue_timing.csv")

    print("Starting timing for LinkedList Get Last operation...")
    linked_list_get_last_times = time_linked_list_get_last()
    plot_results(linked_list_get_last_times, "Linked List Get Last Timing", "Size of Linked List", "Time (seconds)", "linked_list_get_last_timing.png")
    save_results_to_csv(linked_list_get_last_times, "linked_list_get_last_timing.csv")

    print("Starting timing for LinkedList Print All operation...")
    linked_list_print_all_times = time_linked_list_print_all()
    plot_results(linked_list_print_all_times, "Linked List Print All Timing", "Size of Linked List", "Time (seconds)", "linked_list_print_all_timing.png")
    save_results_to_csv(linked_list_print_all_times, "linked_list_print_all_timing.csv")

    print("Starting timing for Max Heap Add operation...")
    max_heap_add_times = time_max_heap_add()
    plot_results(max_heap_add_times, "Max Heap Add Timing", "Size of Heap", "Time (seconds)", "max_heap_add_timing.png")
    save_results_to_csv(max_heap_add_times, "max_heap_add_timing.csv")

    print("Starting timing for BST Search operation...")
    bst_search_times = time_bst_search()
    plot_results(bst_search_times, "BST Search Timing", "Size of BST", "Time (seconds)", "bst_search_timing.png")
    save_results_to_csv(bst_search_times, "bst_search_timing.csv")
