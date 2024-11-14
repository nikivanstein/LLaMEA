import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
    
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
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_solution = improvise(harmony_memory)
            if func(new_solution) < func(harmony_memory[-1]):
                harmony_memory = update_harmony_memory(harmony_memory, new_solution)
        
        return harmony_memory[0]