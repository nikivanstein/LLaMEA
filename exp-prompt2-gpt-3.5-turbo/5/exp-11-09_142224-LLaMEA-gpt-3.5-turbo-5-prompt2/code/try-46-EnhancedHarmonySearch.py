import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound)

    def create_new_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                new_harmony[i] += np.random.normal(0, self.bandwidth)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        self.bandwidth *= 0.99  # Adaptive bandwidth adjustment
        return new_harmony