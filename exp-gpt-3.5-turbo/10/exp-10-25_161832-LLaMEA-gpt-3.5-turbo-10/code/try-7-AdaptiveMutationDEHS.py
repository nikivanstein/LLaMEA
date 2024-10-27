import numpy as np

class AdaptiveMutationDEHS(DiversityEnhancedHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_prob = 0.1

    def __call__(self, func):
        def improvise_harmony(harmony_memory):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.mutation_prob:
                    new_harmony[d] = np.random.uniform(self.lower_bound, self.upper_bound)
                else:
                    idx = np.random.randint(self.budget)
                    new_harmony[d] = harmony_memory[idx, d]
            return new_harmony

        harmony_memory = self.initialize_harmony_memory()
        for _ in range(self.budget):
            new_solution = improvise_harmony(harmony_memory)
            self.update_harmony_memory(harmony_memory, new_solution)

        return harmony_memory[np.argmin([func(h) for h in harmony_memory])]