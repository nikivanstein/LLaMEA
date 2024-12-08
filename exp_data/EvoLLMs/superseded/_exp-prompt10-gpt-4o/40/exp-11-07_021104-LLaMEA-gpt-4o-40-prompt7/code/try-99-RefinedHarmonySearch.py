import numpy as np

class RefinedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Slightly reduced size for faster convergence
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.75  # Adjusted for improved exploration-exploitation balance
        self.pitch_adjustment_rate = 0.3  # Fine-tuned for better stability
        self.bandwidth_reduction = 0.07  # Optimized for better exploitation

    def __call__(self, func):
        evaluations = 0
        # Initial evaluations for harmony memory
        for i in range(min(self.harmony_memory_size, self.budget)):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(new_harmony)
            evaluations += 1

            # Replace worst harmony if new is better
            worst_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_index]:
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

            self._adaptive_memory_adjustment(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = []
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = np.random.choice(self.harmony_memory[:, i])
                if np.random.rand() < self.pitch_adjustment_rate:
                    selected_harmony += (np.random.rand() - 0.5) * self.bandwidth_reduction
                harmony.append(selected_harmony)
            else:
                harmony.append(np.random.uniform(self.lower_bound, self.upper_bound))
        return np.clip(harmony, self.lower_bound, self.upper_bound)

    def _adaptive_memory_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.harmony_memory_rate = 0.75 + 0.2 * np.sin(progress_ratio * np.pi)  # Dynamic rate adjustment
        self.bandwidth_reduction = 0.1 * (1 - progress_ratio)  # Gradual bandwidth reduction