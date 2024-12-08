import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Slightly reduced size for more efficient updates
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9  # Increased for more exploitation
        self.pitch_adjustment_rate = 0.3  # Reduced for stability in exploration
        self.bandwidth_reduction = 0.06  # Reduced for finer adjustments

    def __call__(self, func):
        evaluations = 0
        for i in range(min(self.harmony_memory_size, self.budget)):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(np.clip(new_harmony, self.lower_bound, self.upper_bound))
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                replace_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[replace_index] = new_harmony
                self.harmony_memory_values[replace_index] = new_value

            self._dynamic_memory_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_index = np.random.randint(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[selected_index, i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    direction = np.random.choice([-1, 1])
                    harmony[i] += direction * (np.random.rand() * self.bandwidth_reduction)
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _dynamic_memory_adjustment(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(progress * np.pi / 2)  # Adaptive rate using cosine function
        self.bandwidth_reduction = 0.05 * (1 - progress)  # Gradual bandwidth reduction