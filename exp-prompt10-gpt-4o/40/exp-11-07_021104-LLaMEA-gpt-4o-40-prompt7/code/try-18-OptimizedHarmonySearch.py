import numpy as np

class OptimizedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10  # Reduced size for efficiency
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Higher rate for exploration
        self.pitch_adjustment_rate = 0.25  # Lower rate for stability
        self.bandwidth = 0.2  # Broader initial bandwidth

    def __call__(self, func):
        evaluations = 0
        self.harmony_memory_values = np.apply_along_axis(func, 1, self.harmony_memory)  # Vectorized evaluation
        evaluations += self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.clip(self._create_new_harmony(), self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            worst_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_index]:
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

            self._dynamic_adjustments(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _create_new_harmony(self):
        harmony = np.where(np.random.rand(self.dim) < self.harmony_memory_rate,
                           self.harmony_memory[np.random.randint(self.harmony_memory_size), :],
                           np.random.uniform(self.lower_bound, self.upper_bound, self.dim))
        adjust = np.random.rand(self.dim) < self.pitch_adjustment_rate
        harmony[adjust] += self.bandwidth * (2 * np.random.rand(np.sum(adjust)) - 1)
        return harmony

    def _dynamic_adjustments(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(np.pi * progress)
        self.bandwidth *= 0.95  # Gradual reduction