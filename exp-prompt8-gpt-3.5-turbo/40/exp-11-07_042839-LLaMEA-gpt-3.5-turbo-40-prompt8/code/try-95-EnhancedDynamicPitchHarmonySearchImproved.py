import numpy as np

class EnhancedDynamicPitchHarmonySearchImproved:
    def __init__(self, budget, dim, harmony_memory_size=10, initial_pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.initial_pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, initial_pitch_adjustment_rate, bandwidth

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        pitch_adjustment_rate = np.linspace(self.initial_pitch_adjustment_rate, 0, self.budget)  # Pre-calculate pitch adjustment rates
        
        min_index = np.argmin(harmony_scores)  # Move outside the loop for efficiency
        
        for i in range(self.budget):
            adjustment = np.random.uniform(-pitch_adjustment_rate[i], pitch_adjustment_rate[i], (self.dim,))
            new_harmony = np.clip(adjustment + harmony_memory[min_index], -5.0, 5.0)
            new_score = func(new_harmony)
            
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index] = new_harmony
                harmony_scores[min_index] = new_score
                
        return harmony_memory[np.argmin(harmony_scores)]