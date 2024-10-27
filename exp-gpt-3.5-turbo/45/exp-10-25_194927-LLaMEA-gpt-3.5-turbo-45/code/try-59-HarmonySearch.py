import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = 0.01

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        for _ in range(self.budget):
            new_solution = np.mean(harmony_memory, axis=0) + np.random.uniform(-1, 1, self.dim) * self.bandwidth
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)

            if func(new_solution) < func(harmony_memory[0]):
                harmony_memory[0] = new_solution
        
        return harmony_memory[0]