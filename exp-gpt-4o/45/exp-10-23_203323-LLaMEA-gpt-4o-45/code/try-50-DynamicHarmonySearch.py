import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjustment_rate_initial = 0.3
        self.pitch_adjustment_rate_final = 0.9
        self.bandwidth = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.random_restart_probability = 0.1

    def __call__(self, func):
        np.random.seed(0)
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        harmony_scores = np.apply_along_axis(func, 1, harmony_memory)
        evaluations = self.harmony_memory_size
        best_harmony_index = np.argmin(harmony_scores)
        best_harmony = harmony_memory[best_harmony_index]
        best_score = harmony_scores[best_harmony_index]

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                    if np.random.rand() < self.pitch_adjustment_rate_initial + (
                        (self.pitch_adjustment_rate_final - self.pitch_adjustment_rate_initial) * evaluations / self.budget):
                        new_harmony[i] += self.bandwidth * (2 * np.random.rand() - 1)
                else:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_score = func(new_harmony)
            evaluations += 1

            if new_score < best_score:
                best_score = new_score
                best_harmony = new_harmony.copy()

            worst_index = np.argmax(harmony_scores)
            if new_score < harmony_scores[worst_index]:
                harmony_memory[worst_index] = new_harmony
                harmony_scores[worst_index] = new_score

            if np.random.rand() < self.random_restart_probability:
                restart_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                restart_score = func(restart_harmony)
                evaluations += 1
                if restart_score < best_score:
                    best_score = restart_score
                    best_harmony = restart_harmony.copy()
            
# Usage:
# dynamic_hs = DynamicHarmonySearch(budget=10000, dim=10)
# dynamic_hs(func)