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
        self.harmony_memory_rate = 0.9  # Increased for exploration
        self.dynamic_pitch_adjustment_rate = 0.2  # Dynamic pitch adjustment
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

            if new_value < max(self.harmony_memory_values):
                worst_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

            self._dynamic_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_index = np.random.choice(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[selected_index, i]
                if np.random.rand() < self.dynamic_pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth_reduction
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _dynamic_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.dynamic_pitch_adjustment_rate = 0.25 * np.cos(progress_ratio * np.pi / 2)  # Cosine-based rate adjustment
        self.bandwidth_reduction = 0.09 * (1 - progress_ratio ** 2)  # Quadratic bandwidth modification