import numpy as np

class AdvancedHarmonySearchEnhanced:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])

        for _ in range(self.budget):
            pitch_adjustments = np.abs(harmony_memory - np.mean(harmony_memory, axis=0)) * self.pitch_adjustment_rate
            new_harmony = np.clip(np.random.normal(np.mean(harmony_memory, axis=0), pitch_adjustments), -5.0, 5.0)
            new_score = func(new_harmony)
            min_index = np.argmin(harmony_scores)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index], harmony_scores[min_index] = new_harmony, new_score

        return harmony_memory[np.argmin(harmony_scores)]