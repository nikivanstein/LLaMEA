import numpy as np

class OptimizedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 12  # Reduced size for faster operation
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.harmony_memory_size, np.inf)
        self.harmony_memory_rate = 0.9  # More exploration
        self.pitch_adjustment_rate = 0.3  # Balanced rate
        self.bandwidth = 0.1  # Reduced for finer local search
        
    def __call__(self, func):
        evaluations = 0
        self.harmony_memory_values[:self.harmony_memory_size] = [func(harmony) for harmony in self.harmony_memory]
        evaluations += self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(new_harmony)
            evaluations += 1

            max_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[max_index]:
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value
            
            progress = evaluations / self.budget
            self.harmony_memory_rate = 0.9 - 0.1 * progress
            self.bandwidth *= 0.95  # Gradually reduce bandwidth

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        new_harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                harmony_value = self.harmony_memory[np.random.randint(self.harmony_memory_size), i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    harmony_value += self.bandwidth * (np.random.rand() - 0.5) * 2
            else:
                harmony_value = np.random.uniform(self.lower_bound, self.upper_bound)
            new_harmony[i] = np.clip(harmony_value, self.lower_bound, self.upper_bound)
        return new_harmony