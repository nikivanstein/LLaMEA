import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Reduced size for faster updates
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Higher rate for more exploitation
        self.pitch_adjustment_rate = 0.3  # Adjusted for stability
        self.bandwidth = 0.1  # Static bandwidth for simplicity
        self.evaluations = 0

    def __call__(self, func):
        # Initial evaluations using vectorized operation
        initial_evals = min(self.harmony_memory_size, self.budget)
        self.harmony_memory_values[:initial_evals] = np.apply_along_axis(func, 1, self.harmony_memory[:initial_evals])
        self.evaluations += initial_evals

        while self.evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            self.evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                max_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            # Simplified adaptive memory adjustment
            self._adjust_harmony_memory_rate()

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = self.harmony_memory[np.random.randint(self.harmony_memory_size)].copy()
        for i in range(self.dim):
            if np.random.rand() < self.pitch_adjustment_rate:
                harmony[i] += (np.random.rand() - 0.5) * self.bandwidth
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _adjust_harmony_memory_rate(self):
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(self.evaluations / self.budget * np.pi)