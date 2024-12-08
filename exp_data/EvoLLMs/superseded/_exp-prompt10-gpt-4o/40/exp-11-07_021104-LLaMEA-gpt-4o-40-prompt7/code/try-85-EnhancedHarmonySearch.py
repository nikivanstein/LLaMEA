import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Reduced size for even faster updates
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.inf * np.ones(self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Increased for more exploitation
        self.pitch_adjustment_rate = 0.3  # Further reduced for stability
        self.bandwidth_reduction = 0.07  # Fine-tuned bandwidth adjustment

    def __call__(self, func):
        evaluations = 0
        for i in range(min(self.harmony_memory_size, self.budget)):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                self._replace_worst_harmony(new_harmony, new_value)

            self._simplified_adaptive_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.copy(self.harmony_memory[np.random.randint(self.harmony_memory_size)])
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth_reduction
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _replace_worst_harmony(self, new_harmony, new_value):
        max_index = np.argmax(self.harmony_memory_values)
        self.harmony_memory[max_index] = new_harmony
        self.harmony_memory_values[max_index] = new_value

    def _simplified_adaptive_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.05 * np.cos(progress_ratio * np.pi)  # Simplified rate change
        self.bandwidth_reduction = 0.08 * (1 - progress_ratio)  # Simplified bandwidth reduction