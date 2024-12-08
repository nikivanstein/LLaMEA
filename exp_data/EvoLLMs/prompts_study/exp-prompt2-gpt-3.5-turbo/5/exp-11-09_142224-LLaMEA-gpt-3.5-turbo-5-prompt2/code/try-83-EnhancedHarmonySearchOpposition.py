import numpy as np

class EnhancedHarmonySearchOpposition(EnhancedHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        for _ in range(self.budget - 1):
            self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound) * (1.0 - _ / self.budget)  # Dynamic bandwidth adaptation
            new_harmony = self.create_new_harmony(harmony_memory)
            new_harmony_opposite = 2.0 * self.lower_bound + self.upper_bound - new_harmony  # Opposition-based learning
            new_fitness = func(new_harmony)
            new_fitness_opposite = func(new_harmony_opposite)
            if new_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
            if new_fitness_opposite < harmony_memory_fitness:  # Local search in opposite space
                harmony_memory = new_harmony_opposite
                harmony_memory_fitness = new_fitness_opposite
        return harmony_memory