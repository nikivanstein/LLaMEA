import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.bandwidth = bandwidth

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        for _ in range(self.budget):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            new_score = func(new_harmony)
            min_index = np.argmin(harmony_scores)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index] = new_harmony
                harmony_scores[min_index] = new_score
        return harmony_memory[np.argmin(harmony_scores)]