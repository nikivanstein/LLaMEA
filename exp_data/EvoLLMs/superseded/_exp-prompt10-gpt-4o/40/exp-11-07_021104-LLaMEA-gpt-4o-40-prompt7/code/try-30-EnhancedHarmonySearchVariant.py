import numpy as np

class EnhancedHarmonySearchVariant:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 12  # Smaller size for quicker iterations
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9  # Increased rate for intensified exploration
        self.pitch_adjustment_rate = 0.4  # More frequent pitch adjustments
        self.bandwidth_reduction = 0.05  # Gradually tightening search space

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.harmony_memory_size, self.budget)
        for i in range(initial_evals):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                max_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            self._dynamic_adjustments(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                harmony[i] = self.harmony_memory[np.random.randint(self.harmony_memory_size), i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * 2 * self.bandwidth_reduction
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _dynamic_adjustments(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.05 * np.sin(progress_ratio * np.pi)  # Adaptive rate change
        self.bandwidth_reduction = 0.1 * (1 - progress_ratio)  # Dynamic bandwidth reduction