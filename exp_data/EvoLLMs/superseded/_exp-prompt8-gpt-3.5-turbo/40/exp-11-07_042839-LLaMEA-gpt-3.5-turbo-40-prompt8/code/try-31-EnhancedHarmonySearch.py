import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        for _ in range(self.budget):
            new_harmony = np.clip(np.random.uniform(-self.bandwidth, self.bandwidth, (1, self.dim)) + harmony_memory[np.argmin(harmony_scores)], -5.0, 5.0)
            new_score = func(new_harmony)
            better_indices = harmony_scores > new_score
            harmony_memory[better_indices] = new_harmony
            harmony_scores[better_indices] = new_score
            
        return harmony_memory[np.argmin(harmony_scores)]