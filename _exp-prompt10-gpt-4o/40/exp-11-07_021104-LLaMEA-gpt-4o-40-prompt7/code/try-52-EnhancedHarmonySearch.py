import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Reduced size for quicker adaptation
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9  # Higher base rate for quicker convergence
        self.pitch_adjustment_rate = 0.3  # Balanced adjustment
        self.bandwidth_reduction = 0.1  # Initial bandwidth adjustment

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.harmony_memory_size, self.budget)
        self.harmony_memory_values[:initial_evals] = [func(harmony) for harmony in self.harmony_memory[:initial_evals]]
        evaluations += initial_evals

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(np.clip(new_harmony, self.lower_bound, self.upper_bound))
            evaluations += 1

            max_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[max_index]:
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            self._dynamic_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.where(np.random.rand(self.dim) < self.harmony_memory_rate,
                           self.harmony_memory[np.random.randint(self.harmony_memory_size), np.arange(self.dim)],
                           np.random.uniform(self.lower_bound, self.upper_bound, self.dim))
        pitch_adjust = np.random.rand(self.dim) < self.pitch_adjustment_rate
        harmony[pitch_adjust] += (np.random.rand(np.sum(pitch_adjust)) - 0.5) * self.bandwidth_reduction
        return harmony

    def _dynamic_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.85 + 0.1 * np.cos(progress_ratio * np.pi)  # Cosine for smoother rate change
        self.bandwidth_reduction *= 0.98  # Gradual bandwidth reduction