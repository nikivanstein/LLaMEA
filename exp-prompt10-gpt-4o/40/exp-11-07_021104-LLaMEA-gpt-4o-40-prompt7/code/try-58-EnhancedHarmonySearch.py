import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_size = 10  # Same size for initial simplicity
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.memory_size, self.dim))
        self.memory_values = np.array([np.inf] * self.memory_size)
        self.memory_rate = 0.85  # Unchanged for consistency
        self.pitch_rate = 0.4  # Increased for potentially better adjustments
        self.bandwidth = 0.1  # Adjusted for broader adjustments

    def __call__(self, func):
        evaluations = 0
        for i in range(min(self.memory_size, self.budget)):
            self.memory_values[i] = func(self.memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._create_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            max_index = np.argmax(self.memory_values)
            if new_value < self.memory_values[max_index]:
                self.memory[max_index] = new_harmony
                self.memory_values[max_index] = new_value

            self._update_memory_strategy(evaluations)

        return self.memory[np.argmin(self.memory_values)]

    def _create_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.memory_rate:
                harmony[i] = self.memory[np.random.randint(self.memory_size), i]
                if np.random.rand() < self.pitch_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _update_memory_strategy(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.memory_rate = 0.9 * (1 - np.cos(progress_ratio * np.pi))  # Smoother adaptive rate
        self.bandwidth = 0.2 * (1 - progress_ratio)  # Broader initial bandwidth