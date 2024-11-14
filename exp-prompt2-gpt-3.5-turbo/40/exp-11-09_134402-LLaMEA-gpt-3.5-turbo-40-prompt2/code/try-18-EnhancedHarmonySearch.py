import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, bandwidth=0.01, bandwidth_range=(0.01, 0.1), pitch_adjustment_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.bandwidth_range = bandwidth_range
        self.pitch_adjustment_rate = pitch_adjustment_rate
    
    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        
        def update_harmony_memory(harmony_memory, new_solution):
            harmony_memory = np.vstack((harmony_memory, new_solution))
            harmony_memory = harmony_memory[np.argsort(func(harmony_memory))]
            return harmony_memory[:self.harmony_memory_size]
        
        def improvise(harmony_memory):
            new_solution = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < self.bandwidth:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_solution[i] += np.random.uniform(-1.0, 1.0)
                    new_solution[i] = np.clip(new_solution[i], -5.0, 5.0)
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            self.bandwidth = np.clip(self.bandwidth + np.random.uniform(-0.01, 0.01), *self.bandwidth_range)
            new_solution = improvise(harmony_memory)
            if func(new_solution) < func(harmony_memory[-1]):
                harmony_memory = update_harmony_memory(harmony_memory, new_solution)
        
        return harmony_memory[0]