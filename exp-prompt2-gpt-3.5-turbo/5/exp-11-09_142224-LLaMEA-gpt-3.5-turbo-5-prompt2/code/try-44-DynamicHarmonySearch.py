import numpy as np

class DynamicHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        for _ in range(self.budget - 1):
            new_harmony = self.create_new_harmony(harmony_memory)
            new_fitness = func(new_harmony)
            if new_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
                self.bandwidth *= 1.01  # Increase bandwidth for exploration
            else:
                self.bandwidth *= 0.99  # Decrease bandwidth for exploitation
        return harmony_memory