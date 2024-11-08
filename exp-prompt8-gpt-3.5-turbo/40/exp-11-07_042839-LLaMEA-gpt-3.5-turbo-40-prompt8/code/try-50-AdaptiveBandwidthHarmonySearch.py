import numpy as np

class AdaptiveBandwidthHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, initial_bandwidth=0.01, bandwidth_decay=0.95, bandwidth_min=0.001):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate = budget, dim, harmony_memory_size, pitch_adjustment_rate
        self.bandwidth, self.bandwidth_decay, self.bandwidth_min = initial_bandwidth, bandwidth_decay, bandwidth_min

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        for _ in range(self.budget):
            self.bandwidth = max(self.bandwidth * self.bandwidth_decay, self.bandwidth_min)
            new_harmony = np.clip(np.random.uniform(-self.bandwidth, self.bandwidth, (self.dim,)) + harmony_memory[np.argmin(harmony_scores)], -5.0, 5.0)
            new_score = func(new_harmony)
            min_index = np.argmin(harmony_scores)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index], harmony_scores[min_index] = new_harmony, new_score
                
        return harmony_memory[np.argmin(harmony_scores)]