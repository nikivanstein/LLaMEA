import numpy as np

class ImprovedHarmonySearch(EnhancedHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        global_best_harmony = np.copy(harmony_memory)
        global_best_fitness = harmony_memory_fitness
        for _ in range(self.budget - 1):
            self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound) * (1.0 - _ / self.budget)  # Dynamic bandwidth adaptation
            new_harmony = self.create_new_harmony(harmony_memory)
            new_fitness = func(new_harmony)
            if new_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
            if new_fitness < global_best_fitness:
                global_best_harmony = np.copy(new_harmony)
                global_best_fitness = new_fitness
        return global_best_harmony