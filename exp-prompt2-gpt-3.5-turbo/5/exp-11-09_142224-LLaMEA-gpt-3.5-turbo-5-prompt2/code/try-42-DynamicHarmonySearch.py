import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound)
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        convergence_rate = 0.1
        for _ in range(self.budget - 1):
            new_harmony = self.create_new_harmony(harmony_memory, convergence_rate)
            new_fitness = func(new_harmony)
            if new_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
                if new_fitness < func(harmony_memory):
                    convergence_rate *= 1.2
                else:
                    convergence_rate *= 0.8
        return harmony_memory
    
    def create_new_harmony(self, harmony_memory, convergence_rate):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                new_harmony[i] += np.random.uniform(-self.bandwidth * convergence_rate, self.bandwidth * convergence_rate)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony