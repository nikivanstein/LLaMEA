import numpy as np

class OptimizedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 12  # Reduced size for faster operations
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Increased rate for better exploitation
        self.pitch_adjustment_rate = 0.3  # Slightly lowered pitch rate
        self.bandwidth = 0.2  # Wider initial bandwidth for early exploration

    def __call__(self, func):
        evaluations = 0
        self.harmony_memory_values[:self.harmony_memory_size] = [func(harmony) for harmony in self.harmony_memory]
        evaluations += self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.clip(self._create_new_harmony(), self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            worst_idx = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_idx]:
                self.harmony_memory[worst_idx], self.harmony_memory_values[worst_idx] = new_harmony, new_value

            self._update_parameters(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _create_new_harmony(self):
        return np.array([self._select_harmony_value(i) for i in range(self.dim)])

    def _select_harmony_value(self, index):
        if np.random.rand() < self.harmony_memory_rate:
            value = self.harmony_memory[np.random.randint(self.harmony_memory_size), index]
            if np.random.rand() < self.pitch_adjustment_rate:
                value += self.bandwidth * (np.random.rand() * 2 - 1)
        else:
            value = np.random.uniform(self.lower_bound, self.upper_bound)
        return value

    def _update_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(progress * np.pi)
        self.bandwidth = 0.2 * (1 - progress)  # Smooth reduction of bandwidth