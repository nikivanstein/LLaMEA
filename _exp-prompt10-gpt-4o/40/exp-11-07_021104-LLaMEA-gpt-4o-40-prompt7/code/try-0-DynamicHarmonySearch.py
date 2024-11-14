import numpy as np

class DynamicHarmonySearch:
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
        while evaluations < self.budget:
            if evaluations < self.harmony_memory_size:
                self.harmony_memory_values[evaluations] = func(self.harmony_memory[evaluations])
                evaluations += 1
                continue

            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_rate:
                    random_index = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[random_index, i]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[i] += self.bandwidth * (2 * np.random.rand() - 1)
                else:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)

            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < np.max(self.harmony_memory_values):
                worst_index = np.argmax(self.harmony_memory_values)
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

            self.adaptive_adjustments(evaluations)

        best_index = np.argmin(self.harmony_memory_values)
        return self.harmony_memory[best_index]

    def adaptive_adjustments(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.9 - 0.5 * progress
        self.pitch_adjustment_rate = 0.3 + 0.4 * progress
        self.bandwidth = 0.1 * (1 - progress)