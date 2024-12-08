import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = min(10, max(3, dim))  # Adaptive size based on dimension
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.8
        self.pitch_adjustment_rate = 0.3  # Further reduced for consistent improvements
        self.bandwidth_reduction = 0.05  # More stable bandwidth control
        self.dynamic_pop = True  # Flag for dynamic population size

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

            if self.dynamic_pop and evaluations % (self.budget // 10) == 0:
                self._dynamic_population_adjustment()

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = np.random.randint(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[selected_harmony, i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth_reduction
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _dynamic_population_adjustment(self):
        sorted_indices = np.argsort(self.harmony_memory_values)
        top_half = sorted_indices[:len(sorted_indices) // 2]
        self.harmony_memory = self.harmony_memory[top_half]
        self.harmony_memory_values = self.harmony_memory_values[top_half]
        self.harmony_memory_size = len(self.harmony_memory)
        new_harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory = np.vstack((self.harmony_memory, new_harmony_memory))
        self.harmony_memory_values = np.concatenate((self.harmony_memory_values, np.full(self.harmony_memory_size, np.inf)))