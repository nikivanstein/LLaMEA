import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 12  # Slightly increased for diversity
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Increased for more use of memory
        self.pitch_adjustment_rate = 0.3  # Adjusted for better balance
        self.bandwidth = 0.1  # Constant bandwidth for simplicity

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            if evaluations < self.harmony_memory_size:
                self.harmony_memory_values[evaluations] = func(self.harmony_memory[evaluations])
                evaluations += 1
                continue

            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            worst_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_index]:
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

            self._dynamic_parameter_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.random.choice(self.harmony_memory.flatten(), self.dim)
        if np.random.rand() < self.pitch_adjustment_rate:
            harmony += (np.random.rand(self.dim) - 0.5) * self.bandwidth
        return harmony

    def _dynamic_parameter_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(progress_ratio * np.pi)  # Dynamic rate adjustment