import numpy as np

class AdvancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10  # Reduced size for faster adaptation
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, float('inf'))
        self.harmony_memory_rate = 0.85  # Slightly reduced for balance
        self.pitch_adjustment_rate = 0.5  # Increased for diversity
        self.bandwidth = 0.1  # Consistent bandwidth for efficient search

    def __call__(self, func):
        evals = 0
        for i in range(min(self.harmony_memory_size, self.budget)):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evals += 1

        while evals < self.budget:
            new_harmony = self._create_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evals += 1

            if new_value < np.max(self.harmony_memory_values):
                max_idx = np.argmax(self.harmony_memory_values)
                self.harmony_memory[max_idx] = new_harmony
                self.harmony_memory_values[max_idx] = new_value

            self._adjust_parameters(evals)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _create_harmony(self):
        harmony = np.random.choice(self.harmony_memory.flatten(), self.dim).reshape(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                harmony[i] += (np.random.rand() - 0.5) * 2 * self.bandwidth
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _adjust_parameters(self, evals):
        progress = evals / self.budget
        self.harmony_memory_rate = 0.8 + 0.1 * np.cos(progress * np.pi / 2)  # Adaptive rate