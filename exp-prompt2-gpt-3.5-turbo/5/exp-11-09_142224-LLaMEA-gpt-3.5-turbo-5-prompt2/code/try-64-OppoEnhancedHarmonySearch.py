import numpy as np

class OppoEnhancedHarmonySearch(EnhancedHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        for _ in range(self.budget - 1):
            self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound) * (1.0 - _ / self.budget)  # Dynamic bandwidth adaptation
            new_harmony = self.create_new_harmony(harmony_memory)
            new_fitness = func(new_harmony)

            # Opposition-based learning
            oppo_harmony = 2 * np.mean(self.upper_bound) - new_harmony
            oppo_fitness = func(oppo_harmony)
            if oppo_fitness < harmony_memory_fitness:
                harmony_memory = oppo_harmony
                harmony_memory_fitness = oppo_fitness
            else:
                if new_fitness < harmony_memory_fitness:
                    harmony_memory = new_harmony
                    harmony_memory_fitness = new_fitness
        return harmony_memory