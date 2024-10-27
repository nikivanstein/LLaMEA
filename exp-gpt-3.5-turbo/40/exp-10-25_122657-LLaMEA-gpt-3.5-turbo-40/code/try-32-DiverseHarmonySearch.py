# import numpy as np

class DiverseHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def update_harmony_memory(self, harmony_memory, fitness):
        best_idx = np.argmin(fitness)
        new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        harmony_memory[best_idx] = new_solution
        return harmony_memory