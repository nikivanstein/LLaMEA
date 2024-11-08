import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_size = 8  # Reduced size for quicker convergence
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.memory_size, self.dim))
        self.memory_values = np.inf * np.ones(self.memory_size)
        self.memory_rate = 0.9  # Increased for exploration
        self.adjustment_rate = 0.3  # Balanced rate for pitch adjustment
        self.dynamic_bandwidth = 0.05  # Improved dynamic bandwidth

    def __call__(self, func):
        evaluations = 0
        for i in range(min(self.memory_size, self.budget)):
            self.memory_values[i] = func(self.memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            max_value_index = np.argmax(self.memory_values)
            if new_value < self.memory_values[max_value_index]:
                self.memory[max_value_index] = new_harmony
                self.memory_values[max_value_index] = new_value

            self._dynamic_adjustment(evaluations)

        return self.memory[np.argmin(self.memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.memory_rate:
                harmony[i] = np.random.choice(self.memory[:, i])
                if np.random.rand() < self.adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.dynamic_bandwidth
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _dynamic_adjustment(self, evaluations):
        progress = evaluations / self.budget
        self.memory_rate = 0.85 + 0.10 * np.cos(progress * np.pi)  # Cosine-based adjustment
        self.dynamic_bandwidth = 0.06 * (1 - progress)  # Gradual bandwidth reduction