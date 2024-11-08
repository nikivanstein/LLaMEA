import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.dynamic_memory_size = min(20, dim + 5)  # Dynamic size based on dimension
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.dynamic_memory_size, self.dim))
        self.memory_values = np.full(self.dynamic_memory_size, np.inf)
        self.memory_rate = 0.9  # Higher initial exploration
        self.pitch_rate = 0.25  # Reduced for better local search
        self.bandwidth = 0.1  # Slightly increased for initial diversity

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.dynamic_memory_size, self.budget)
        for i in range(initial_evals):
            self.memory_values[i] = func(self.memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._create_harmony()
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.memory_values):
                max_index = np.argmax(self.memory_values)
                self.memory[max_index] = new_harmony
                self.memory_values[max_index] = new_value

            self._adjust_parameters(evaluations)

        return self.memory[np.argmin(self.memory_values)]

    def _create_harmony(self):
        harmony = np.array([
            self._select_memory(i) if np.random.rand() < self.memory_rate else np.random.uniform(self.lower_bound, self.upper_bound)
            for i in range(self.dim)
        ])
        return harmony

    def _select_memory(self, index):
        selected_harmony = np.random.randint(self.dynamic_memory_size)
        value = self.memory[selected_harmony, index]
        if np.random.rand() < self.pitch_rate:
            value += (np.random.uniform(-1, 1) * self.bandwidth)
        return value

    def _adjust_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.memory_rate = 0.85 + 0.1 * progress  # Adaptive rate increase
        self.pitch_rate = 0.2 * (1 - progress)  # Adaptive pitch reduction
        self.bandwidth = 0.05 * (1 - progress)  # Bandwidth reduction over time