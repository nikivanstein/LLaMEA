import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 12  # Slightly increased for better diversity
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.75  # Adjusted for improved balance
        self.pitch_adjustment_rate = 0.30  # Further reduced for enhanced solution stability
        self.bandwidth_reduction = 0.05  # Reduced for finer adjustments

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.harmony_memory_size, self.budget)
        for i in range(initial_evals):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(np.clip(new_harmony, self.lower_bound, self.upper_bound))
            evaluations += 1

            if new_value < np.median(self.harmony_memory_values):  # Median strategy for more robust replacement
                worst_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

            if evaluations % (self.budget // 10) == 0:  # Strategic memory refresh
                self._refresh_memory(func)

            self._adaptive_memory_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)  # Simplified generation
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = np.random.choice(self.harmony_memory_size, 2, replace=False)
                harmony[i] = np.mean(self.harmony_memory[selected_harmony, i])  # Averaged selection for diversity

        return harmony

    def _adaptive_memory_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.7 + 0.2 * np.cos(progress_ratio * np.pi)  # Adaptive rate change using cosine
        self.bandwidth_reduction = 0.05 * np.sqrt(1 - progress_ratio)  # Smoother bandwidth reduction

    def _refresh_memory(self, func):
        for i in range(self.harmony_memory_size // 4):  # Refresh a quarter of the memory
            random_idx = np.random.randint(self.harmony_memory_size)
            self.harmony_memory[random_idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            self.harmony_memory_values[random_idx] = func(self.harmony_memory[random_idx])