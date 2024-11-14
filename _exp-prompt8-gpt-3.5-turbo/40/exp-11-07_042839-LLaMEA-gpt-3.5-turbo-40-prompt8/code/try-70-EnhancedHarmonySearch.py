import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01, max_unchanged_iter=50):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth, self.max_unchanged_iter = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth, max_unchanged_iter

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        unchanged_counter = 0
        best_index = np.argmin(harmony_scores)
        best_score = harmony_scores[best_index]
        
        for _ in range(self.budget):
            new_harmony = np.clip(np.random.uniform(-self.bandwidth, self.bandwidth, (self.dim,)) + harmony_memory[best_index], -5.0, 5.0)
            new_score = func(new_harmony)
            
            if new_score < best_score:
                harmony_memory[best_index], harmony_scores[best_index] = new_harmony, new_score
                unchanged_counter = 0
                best_index = np.argmin(harmony_scores)
                best_score = harmony_scores[best_index]
            else:
                unchanged_counter += 1
            
            if unchanged_counter >= self.max_unchanged_iter:
                break
                
        return harmony_memory[best_index]