import numpy as np

class SimplifiedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 10  # Smaller memory size for faster updates
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Higher rate for intensified exploitation
        self.pitch_adjustment_rate = 0.4  # Balanced pitch adjustment
        self.bandwidth = 0.2  # Initial bandwidth

    def __call__(self, func):
        evaluations = 0
        for i in range(min(self.harmony_memory_size, self.budget)):
            self.harmony_memory_values[i] = func(self.harmony_memory[i])
            evaluations += 1

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(new_harmony)
            evaluations += 1

            max_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[max_index]:
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            self._update_parameters(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        new_harmony = []
        for index in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                harmony_value = self.harmony_memory[np.random.randint(self.harmony_memory_size), index]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony_value += self.bandwidth * (np.random.rand() - 0.5)
            else:
                harmony_value = np.random.uniform(self.lower_bound, self.upper_bound)
            new_harmony.append(harmony_value)
        return np.clip(new_harmony, self.lower_bound, self.upper_bound)

    def _update_parameters(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.9 - 0.2 * progress  # Gradually decreases
        self.bandwidth *= 0.95  # Reduce bandwidth progressively