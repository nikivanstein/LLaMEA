import numpy as np

class DiversityEnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def improvise_harmony(harmony_memory):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < 0.5:
                    new_harmony[d] = np.random.uniform(self.lower_bound, self.upper_bound)
                else:
                    idx = np.random.randint(self.budget)
                    new_harmony[d] = harmony_memory[idx, d]
            return new_harmony

        def update_harmony_memory(harmony_memory, new_solution):
            worst_idx = np.argmax([func(h) for h in harmony_memory])
            if func(new_solution) < func(harmony_memory[worst_idx]):
                harmony_memory[worst_idx] = new_solution

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_solution = improvise_harmony(harmony_memory)
            update_harmony_memory(harmony_memory, new_solution)

        return harmony_memory[np.argmin([func(h) for h in harmony_memory])]