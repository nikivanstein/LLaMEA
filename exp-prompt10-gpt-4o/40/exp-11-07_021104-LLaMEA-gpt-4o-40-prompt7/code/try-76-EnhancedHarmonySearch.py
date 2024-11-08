import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Smaller memory size for faster processing
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Increased for broader memory consideration
        self.pitch_adjustment_rate = 0.3  # Fine-tuned for better stability
        self.bandwidth_reduction = 0.1  # Adjusted bandwidth for efficient search

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
            worst_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_index]:
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value
            self._dynamic_adjustments(evaluations)
        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = self.harmony_memory[np.random.randint(self.harmony_memory_size), i]
                harmony[i] = selected_harmony
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth_reduction
        return harmony

    def _dynamic_adjustments(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(progress_ratio * np.pi)  # Dynamic rate with cosine modulation
        self.bandwidth_reduction = 0.1 * (1 - progress_ratio**2)  # Quadratic reduction for finer tuning