import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 12  # Slightly increased for diversity
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)  # Simplified array initialization
        self.memory_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.4
        self.bandwidth = 0.1

    def __call__(self, func):
        evaluations = 0
        self.harmony_memory_values[:self.harmony_memory_size] = [func(harmony) for harmony in self.harmony_memory]
        evaluations += self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            worst_idx = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                self.harmony_memory_values[worst_idx] = new_value

            self._update_parameters(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.memory_consideration_rate:
                chosen = self.harmony_memory[np.random.randint(self.harmony_memory_size), i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    chosen += (np.random.rand() - 0.5) * self.bandwidth
                harmony[i] = chosen
        return harmony

    def _update_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.memory_consideration_rate = 0.85 + 0.1 * (1 - progress)
        self.bandwidth *= 0.95  # Gradual reduction for finer search