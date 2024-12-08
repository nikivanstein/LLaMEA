import numpy as np

class OptimizedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10  # Reduced size for faster adaptation
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.inf * np.ones(self.harmony_memory_size)
        self.harmony_memory_rate = 0.85  # Adjusted rate for balanced exploration
        self.pitch_adjustment_rate = 0.3  # Less frequent pitch adjustments
        self.bandwidth_reduction = 0.08  # Moderate tightening for controlled search space

    def __call__(self, func):
        evaluations = 0
        for i in range(self.harmony_memory_size):
            if evaluations < self.budget:
                self.harmony_memory_values[i] = func(self.harmony_memory[i])
                evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            max_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[max_index]:
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            self._dynamic_adjustments(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        draw = np.random.rand(self.dim)
        pitch = np.random.rand(self.dim) < self.pitch_adjustment_rate
        memory_indices = np.random.randint(self.harmony_memory_size, size=self.dim)
        random_values = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        harmony = np.where(draw < self.harmony_memory_rate, 
                           self.harmony_memory[memory_indices, np.arange(self.dim)], 
                           random_values)
        harmony += np.where(pitch, (np.random.rand(self.dim) - 0.5) * 2 * self.bandwidth_reduction, 0)
        return harmony

    def _dynamic_adjustments(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.8 + 0.1 * np.cos(progress_ratio * np.pi / 2)  # Periodic rate adjustment
        self.bandwidth_reduction = 0.08 * (1 - progress_ratio)  # Linearly decreasing bandwidth