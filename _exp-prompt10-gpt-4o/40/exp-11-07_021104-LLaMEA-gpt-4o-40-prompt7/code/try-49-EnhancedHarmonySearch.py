import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Reduced size for faster convergence
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9  # Increased for exploration benefits
        self.pitch_adjustment_rate = 0.3  # Balanced for stability
        self.bandwidth_initial = 0.1

    def __call__(self, func):
        evaluations = 0
        for i in range(self.harmony_memory_size):
            if evaluations < self.budget:
                self.harmony_memory_values[i] = func(self.harmony_memory[i])
                evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony(evaluations / self.budget)
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            worst_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_index]:
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self, progress_ratio):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = np.random.randint(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[selected_harmony, i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth_initial * (1 - progress_ratio)
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony