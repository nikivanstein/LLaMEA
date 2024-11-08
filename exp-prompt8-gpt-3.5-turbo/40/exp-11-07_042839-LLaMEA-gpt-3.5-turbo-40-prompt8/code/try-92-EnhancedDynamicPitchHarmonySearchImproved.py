import numpy as np

class EnhancedDynamicPitchHarmonySearchImproved:
    def __init__(self, budget, dim, harmony_memory_size=10, initial_pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.initial_pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, initial_pitch_adjustment_rate, bandwidth

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        pitch_adjustment_rate = np.linspace(self.initial_pitch_adjustment_rate, 0, self.budget)  # Pre-calculate pitch adjustment rates
        
        for i in range(self.budget):
            new_harmony = np.clip(np.random.uniform(-pitch_adjustment_rate[i], pitch_adjustment_rate[i], (self.dim,)) + harmony_memory[np.argmin(harmony_scores)], -5.0, 5.0)
            new_score = func(new_harmony)
            min_index = np.argmin(harmony_scores)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index] = new_harmony
                harmony_scores[min_index] = new_score
        
        # Introduce elitism to preserve best solutions found
        best_index = np.argmin(harmony_scores)
        best_solution = harmony_memory[best_index]
        best_score = harmony_scores[best_index]

        return best_solution