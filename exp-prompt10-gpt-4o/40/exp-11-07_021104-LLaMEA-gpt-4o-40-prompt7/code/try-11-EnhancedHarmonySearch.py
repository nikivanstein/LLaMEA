import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 15  # Reduced size for faster convergence
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.85  # Modified rate for better exploration
        self.pitch_adjustment_rate = 0.35  # Adjusted pitch rate for balance
        self.bandwidth = 0.15  # Increased bandwidth for wider search space

    def __call__(self, func):
        evaluations = 0
        for i in range(min(self.harmony_memory_size, self.budget)):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = np.array([self._generate_harmony(i) for i in range(self.dim)])
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            max_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[max_index]:
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            self._adaptive_adjustments(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self, index):
        if np.random.rand() < self.harmony_memory_rate:
            harmony_value = self.harmony_memory[np.random.randint(self.harmony_memory_size), index]
            if np.random.rand() < self.pitch_adjustment_rate:
                harmony_value += self.bandwidth * (np.random.rand() - 0.5) * 2
        else:
            harmony_value = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony_value

    def _adaptive_adjustments(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.8 + 0.1 * np.cos(progress * np.pi)  # Dynamic adjustment
        self.bandwidth = 0.15 * (1 - progress)  # Gradually reduce bandwidth