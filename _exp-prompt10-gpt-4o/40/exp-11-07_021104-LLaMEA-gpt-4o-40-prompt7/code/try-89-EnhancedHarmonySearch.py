import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Slightly reduced for efficiency
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([float('inf')] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Increased for exploration-exploitation balance
        self.pitch_adjustment_rate = 0.25  # Reduced for stability
        self.bandwidth = 0.05  # Fixed bandwidth for consistency

    def __call__(self, func):
        evaluations = 0
        for i in range(self.harmony_memory_size):
            if evaluations >= self.budget:
                break
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                max_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                harmony[i] = np.random.choice(self.harmony_memory[:, i])
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony