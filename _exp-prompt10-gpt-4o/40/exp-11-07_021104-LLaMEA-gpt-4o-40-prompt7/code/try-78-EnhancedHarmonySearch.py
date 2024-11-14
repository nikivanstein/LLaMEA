import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Smaller memory size for quicker adaptation
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9  # Increased for more exploration
        self.pitch_adjustment_rate = 0.3  # Balanced rate for stability
        self.adaptive_bandwidth = 0.1  # Adaptive bandwidth for better exploration

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

            worse_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worse_index]:
                self.harmony_memory[worse_index] = new_harmony
                self.harmony_memory_values[worse_index] = new_value

            self._adaptive_step_size(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_indices = np.argsort(self.harmony_memory_values)
                selected_harmony = np.random.choice(selected_indices[:self.harmony_memory_size // 2])
                harmony[i] = self.harmony_memory[selected_harmony, i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.adaptive_bandwidth
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _adaptive_step_size(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(progress_ratio * np.pi)  # Smooth rate adaptation
        self.adaptive_bandwidth = 0.1 * (1 - progress_ratio)  # Dynamic step-size adjustment