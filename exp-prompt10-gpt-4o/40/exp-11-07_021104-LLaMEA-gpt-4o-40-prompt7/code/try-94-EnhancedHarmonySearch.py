import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.dynamic_bandwidth = 0.1

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

            worst_idx = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                self.harmony_memory_values[worst_idx] = new_value

            self._update_parameters(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        rand_values = np.random.rand(self.dim)
        harmony = np.where(rand_values < self.harmony_memory_rate,
                           self.harmony_memory[np.random.randint(self.harmony_memory_size), np.arange(self.dim)],
                           np.random.uniform(self.lower_bound, self.upper_bound, self.dim))
        pitch_adjustment_mask = np.random.rand(self.dim) < self.pitch_adjustment_rate
        harmony += pitch_adjustment_mask * ((np.random.rand(self.dim) - 0.5) * self.dynamic_bandwidth)
        return harmony

    def _update_parameters(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.9 * (1 - 0.3 * np.cos(progress_ratio * np.pi))
        self.dynamic_bandwidth = 0.1 * np.exp(-progress_ratio)