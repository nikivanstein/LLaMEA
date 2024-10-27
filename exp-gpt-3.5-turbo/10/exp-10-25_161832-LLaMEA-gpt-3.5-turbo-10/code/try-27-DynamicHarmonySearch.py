import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pitch_adjustment_rate = 0.3

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def improvise_harmony(harmony_memory, iteration):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[d] = np.random.uniform(self.lower_bound, self.upper_bound)
                else:
                    idx = np.random.randint(self.budget)
                    new_harmony[d] = harmony_memory[idx, d]
            return new_harmony

        def update_harmony_memory(harmony_memory, new_solution, func):
            worst_idx = np.argmax([func(h) for h in harmony_memory])
            if func(new_solution) < func(harmony_memory[worst_idx]):
                harmony_memory[worst_idx] = new_solution

        harmony_memory = initialize_harmony_memory()
        for i in range(self.budget):
            new_solution = improvise_harmony(harmony_memory, i/self.budget)
            update_harmony_memory(harmony_memory, new_solution, func)

        return harmony_memory[np.argmin([func(h) for h in harmony_memory])]