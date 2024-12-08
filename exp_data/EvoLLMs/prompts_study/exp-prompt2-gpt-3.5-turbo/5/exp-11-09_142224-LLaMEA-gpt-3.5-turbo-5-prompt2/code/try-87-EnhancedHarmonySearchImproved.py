import numpy as np

class EnhancedHarmonySearchImproved(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        for _ in range(self.budget - 1):
            self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound) * (1.0 - _ / self.budget)  # Dynamic bandwidth adaptation
            new_harmony = self.create_new_harmony(harmony_memory)
            new_fitness = func(new_harmony)
            if new_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
            else:
                new_harmony = self.adjust_pitch(new_harmony, harmony_memory, self.bandwidth)  # Improved pitch adjustment
                new_fitness = func(new_harmony)
                if new_fitness < harmony_memory_fitness:
                    harmony_memory = new_harmony
                    harmony_memory_fitness = new_fitness
        return harmony_memory