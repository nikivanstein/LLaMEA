import numpy as np

class DynamicBandwidthHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, initial_bandwidth=0.01, bandwidth_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.initial_bandwidth = initial_bandwidth
        self.bandwidth_decay = bandwidth_decay
    
    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        
        def update_harmony_memory(harmony_memory, new_solution):
            harmony_memory = np.vstack((harmony_memory, new_solution))
            harmony_memory = harmony_memory[np.argsort(func(harmony_memory))]
            return harmony_memory[:self.harmony_memory_size]
        
        def improvise(harmony_memory, bandwidth):
            new_solution = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < bandwidth:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        bandwidth = self.initial_bandwidth
        for _ in range(self.budget):
            new_solution = improvise(harmony_memory, bandwidth)
            if func(new_solution) < func(harmony_memory[-1]):
                harmony_memory = update_harmony_memory(harmony_memory, new_solution)
            bandwidth *= self.bandwidth_decay
        
        return harmony_memory[0]