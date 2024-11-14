import numpy as np

class HarmonySearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        harmony_memory_fitness = np.array([func(harmony) for harmony in harmony_memory])
        
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_harmony_fitness = func(new_harmony)
            
            if new_harmony_fitness < harmony_memory_fitness.max():
                index = harmony_memory_fitness.argmax()
                harmony_memory[index] = new_harmony
                harmony_memory_fitness[index] = new_harmony_fitness
        
        best_harmony = harmony_memory[np.argmin(harmony_memory_fitness)]
        
        # Introducing a random local search step
        local_search_point = np.clip(best_harmony + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
        local_search_fitness = func(local_search_point)
        
        if local_search_fitness < harmony_memory_fitness.min():
            best_harmony = local_search_point
        
        return best_harmony