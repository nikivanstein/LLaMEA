import numpy as np

class DynamicBandwidthHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, initial_bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate = budget, dim, harmony_memory_size, pitch_adjustment_rate
        self.bandwidth = initial_bandwidth

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        for _ in range(self.budget):
            new_harmony = np.clip(np.random.uniform(-self.bandwidth, self.bandwidth, (self.dim,)) + harmony_memory[np.argmin(harmony_scores)], -5.0, 5.0)
            new_score = func(new_harmony)
            min_index = np.argmin(harmony_scores)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index], harmony_scores[min_index] = new_harmony, new_score
                improvement_rate = (harmony_scores[min_index] - new_score) / harmony_scores[min_index]
                self.bandwidth *= 1.1 if improvement_rate > 0.8 else 0.9  # Dynamic bandwidth adjustment

        return harmony_memory[np.argmin(harmony_scores)]