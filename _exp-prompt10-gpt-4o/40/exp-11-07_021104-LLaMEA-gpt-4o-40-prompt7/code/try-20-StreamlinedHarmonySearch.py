import numpy as np

class StreamlinedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10  # Reduced further for faster convergence
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.apply_along_axis(func, 1, self.harmony_memory)
        self.harmony_memory_rate = 0.9  # Increased for better exploration
        self.pitch_adjustment_rate = 0.3  # Slightly reduced for balance
        self.bandwidth = 0.2  # Adjusted for search space exploration

    def __call__(self, func):
        evaluations = self.harmony_memory_size
        while evaluations < self.budget:
            new_harmony = self._generate_new_harmony()
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                max_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            self._dynamic_parameters(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_new_harmony(self):
        new_harmony = self.harmony_memory[np.random.choice(self.harmony_memory_size, self.dim)]
        adjust_mask = np.random.rand(self.dim) < self.pitch_adjustment_rate
        new_harmony[adjust_mask] += self.bandwidth * (np.random.rand(np.sum(adjust_mask)) - 0.5) * 2
        return np.clip(new_harmony, self.lower_bound, self.upper_bound)

    def _dynamic_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.05 * np.cos(progress * np.pi)  # Refined dynamic adjustment
        self.bandwidth *= (1 - progress)  # Gradually reduce bandwidth