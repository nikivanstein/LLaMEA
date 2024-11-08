import numpy as np

class OptimizedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 20
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.bandwidth = 0.1

    def __call__(self, func):
        evaluations = 0
        initial_eval_count = min(self.harmony_memory_size, self.budget)
        for i in range(initial_eval_count):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
        evaluations += initial_eval_count

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            indices = np.random.choice(self.harmony_memory_size, self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_rate:
                    new_harmony[i] = self.harmony_memory[indices[i], i]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[i] += self.bandwidth * (2 * np.random.rand() - 1)
                else:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            max_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[max_index]:
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            self.adaptive_adjustments(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def adaptive_adjustments(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.95 - 0.5 * progress
        self.pitch_adjustment_rate = 0.25 + 0.5 * progress
        self.bandwidth = 0.1 * (1 - progress)