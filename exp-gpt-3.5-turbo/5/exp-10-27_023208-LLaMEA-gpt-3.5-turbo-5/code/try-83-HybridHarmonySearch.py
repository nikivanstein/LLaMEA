import numpy as np

class HybridHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        
        for _ in range(self.budget):
            new_harmony = self._improvise(harmony_memory)
            harmony_memory = self._update_memory(harmony_memory, new_harmony, func)
        
        best_solution = harmony_memory[np.argmin([func(sol) for sol in harmony_memory])]
        return best_solution

    def _improvise(self, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < self.pitch_adjustment_rate:
                new_harmony[:, i] = np.random.uniform(-5.0, 5.0, self.harmony_memory_size)
        return new_harmony

    def _update_memory(self, harmony_memory, new_harmony, func):
        combined_memory = np.vstack([harmony_memory, new_harmony])
        sorted_indices = np.argsort([func(sol) for sol in combined_memory])
        return combined_memory[sorted_indices[:self.harmony_memory_size]]

hybrid_harmony_search = HybridHarmonySearch(budget=1000, dim=10)

# Testing the algorithm on the BBOB test suite of 24 noiseless functions to evaluate its performance.