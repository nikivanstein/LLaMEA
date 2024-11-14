import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def create_new_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                perturbation = np.random.uniform(-self.bandwidth, self.bandwidth)
                if np.random.rand() < 0.1:  # 5.0% change: Adjust selection probability
                    new_harmony[i] = harmony_memory[i] + perturbation
                else:
                    new_harmony[i] += perturbation
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony