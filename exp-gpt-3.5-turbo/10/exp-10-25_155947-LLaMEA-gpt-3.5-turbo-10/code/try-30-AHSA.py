import numpy as np

class AHSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.pitch_adjustment_rate = 0.3
        self.bandwidth = 0.01
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lb, self.ub, (self.harmony_memory_size, self.dim))

        harmony_memory = initialize_harmony_memory()
        best_solution = harmony_memory[np.argmin([func(harmony) for harmony in harmony_memory])]

        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = best_solution + np.random.uniform(-self.bandwidth, self.bandwidth, (self.dim,))
            new_harmony = np.clip(new_harmony, self.lb, self.ub)

            if func(new_harmony) < func(best_solution):
                best_solution = new_harmony

            random_index = np.random.randint(self.harmony_memory_size)
            if func(new_harmony) < func(harmony_memory[random_index]):
                harmony_memory[random_index] = new_harmony

        return best_solution