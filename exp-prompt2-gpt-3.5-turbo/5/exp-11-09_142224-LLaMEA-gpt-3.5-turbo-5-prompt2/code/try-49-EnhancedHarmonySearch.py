import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound)
    
    def create_new_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        # Dynamic adjustment of bandwidth
        self.bandwidth *= 0.995  # 5% reduction in bandwidth
        return new_harmony