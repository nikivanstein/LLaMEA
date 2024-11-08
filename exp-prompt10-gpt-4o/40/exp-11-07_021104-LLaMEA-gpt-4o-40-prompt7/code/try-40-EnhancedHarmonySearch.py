import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.85
        self.pitch_adjustment_rate = 0.35
        self.bandwidth_reduction = 0.08

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.harmony_memory_size, self.budget)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(func, self.harmony_memory[i]) for i in range(initial_evals)]
            for i, future in enumerate(futures):
                self.harmony_memory_values[i] = future.result()
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

            self._dynamic_parameter_tuning(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = np.random.randint(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[selected_harmony, i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += np.random.uniform(-self.bandwidth_reduction, self.bandwidth_reduction)
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _dynamic_parameter_tuning(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.7 + 0.2 * np.cos(progress_ratio * np.pi)
        self.bandwidth_reduction = 0.1 * np.exp(-progress_ratio)