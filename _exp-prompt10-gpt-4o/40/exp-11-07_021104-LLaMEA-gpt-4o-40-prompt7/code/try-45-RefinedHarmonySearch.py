import numpy as np

class RefinedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Reduced size for quicker adaptation
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.75  # Further reduced to emphasize exploration
        self.pitch_adjustment_rate = 0.25  # Lowered for finer tuning
        self.bandwidth_adjustment = 0.05  # Streamlined bandwidth

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.harmony_memory_size, self.budget)
        for i in range(initial_evals):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                max_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value
                self._local_search(max_index, func)

            self._adaptive_memory_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = np.random.randint(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[selected_harmony, i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.bandwidth_adjustment
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return np.clip(harmony, self.lower_bound, self.upper_bound)

    def _local_search(self, index, func):
        step_size = 0.1  # Small step for local refinement
        original_value = self.harmony_memory_values[index]
        for i in range(self.dim):
            trial_harmony = self.harmony_memory[index].copy()
            trial_harmony[i] += step_size * (np.random.rand() - 0.5)
            trial_harmony = np.clip(trial_harmony, self.lower_bound, self.upper_bound)
            trial_value = func(trial_harmony)
            if trial_value < original_value:
                self.harmony_memory[index, i] = trial_harmony[i]
                self.harmony_memory_values[index] = trial_value
                original_value = trial_value

    def _adaptive_memory_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.7 + 0.1 * np.sin(progress_ratio * np.pi)  # Modified adaptive rate change
        self.bandwidth_adjustment = 0.05 * (1 - progress_ratio)  # Adjusted bandwidth adaptation