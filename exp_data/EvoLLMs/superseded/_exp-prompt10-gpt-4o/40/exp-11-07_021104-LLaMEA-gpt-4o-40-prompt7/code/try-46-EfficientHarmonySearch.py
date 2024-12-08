import numpy as np

class EfficientHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0])
        self.harmony_memory_size = 8  # Slightly reduced for enhanced efficiency
        self.harmony_memory = np.random.uniform(self.bounds[0], self.bounds[1], (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9  # Increased for better exploration
        self.pitch_adjustment_rate = 0.3  # Lowered for stability
        self.bandwidth = 0.05  # Fixed bandwidth for simplicity

    def __call__(self, func):
        evaluations = 0
        self.harmony_memory_values[:self.harmony_memory_size] = [func(h) for h in self.harmony_memory]
        evaluations += self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.bounds[0], self.bounds[1])
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                self._update_memory(new_harmony, new_value)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        if np.random.rand() < self.harmony_memory_rate:
            idx = np.random.choice(self.harmony_memory_size, self.dim, replace=True)
            harmony = self.harmony_memory[idx, np.arange(self.dim)]
            adjustments = (np.random.rand(self.dim) - 0.5) * self.bandwidth
            harmony += np.where(np.random.rand(self.dim) < self.pitch_adjustment_rate, adjustments, 0)
        else:
            harmony = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        return harmony

    def _update_memory(self, new_harmony, new_value):
        worst_idx = np.argmax(self.harmony_memory_values)
        self.harmony_memory[worst_idx] = new_harmony
        self.harmony_memory_values[worst_idx] = new_value