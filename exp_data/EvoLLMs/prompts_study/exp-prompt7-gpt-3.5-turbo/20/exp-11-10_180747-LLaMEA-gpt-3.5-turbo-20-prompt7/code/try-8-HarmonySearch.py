import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        harmony_memory_fitness = np.array([func(harmony) for harmony in harmony_memory])
        
        band_width = 5.0
        for _ in range(self.budget):
            new_harmony = np.random.uniform(harmony_memory.min(axis=0) - band_width, harmony_memory.max(axis=0) + band_width, self.dim)
            new_harmony_fitness = func(new_harmony)
            
            if new_harmony_fitness < harmony_memory_fitness.max():
                index = harmony_memory_fitness.argmax()
                harmony_memory[index] = new_harmony
                harmony_memory_fitness[index] = new_harmony_fitness
                band_width *= 0.95  # Dynamic adjustment of the search space
                
        best_harmony = harmony_memory[np.argmin(harmony_memory_fitness)]
        return best_harmony