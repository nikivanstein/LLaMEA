import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth
        self.random_offsets = np.random.uniform(-self.bandwidth, self.bandwidth, (self.budget, self.dim))

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        for i in range(self.budget):
            min_index = np.argmin(harmony_scores)
            new_harmony = np.clip(self.random_offsets[i] + harmony_memory[min_index], -5.0, 5.0)
            new_score = func(new_harmony)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index], harmony_scores[min_index] = new_harmony, new_score
                
        return harmony_memory[np.argmin(harmony_scores)]