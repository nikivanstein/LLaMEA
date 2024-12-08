import numpy as np

class AdaptiveDifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound, self.upper_bound = -5.0, 5.0
        self.harmony_memory_size = 8  # Reduced for quicker convergence
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.8
        self.mutation_factor = 0.5  # New component for differential mutation
        self.bandwidth_reduction = 0.1

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.harmony_memory_size, self.budget)
        for i in range(initial_evals):
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

            self._adaptive_memory_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = np.random.randint(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[selected_harmony, i]
            else:
                indices = np.random.choice(self.harmony_memory_size, 3, replace=False)
                harmony[i] = self.harmony_memory[indices[0], i] + self.mutation_factor * (self.harmony_memory[indices[1], i] - self.harmony_memory[indices[2], i])
        return harmony

    def _adaptive_memory_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.75 + 0.1 * np.cos(progress_ratio * np.pi)  # Adaptive rate with cosine modulation
        self.bandwidth_reduction *= 0.95  # Gradual bandwidth reduction