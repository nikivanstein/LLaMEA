# import numpy as np

class CustomMetaheuristicAlgorithm:
    def __init__(self, budget, dim, harmony_memory_size=20, hmcr=0.9, par=0.3, mw=0.1, cr=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = hmcr
        self.par = par
        self.mw = mw
        self.cr = cr

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def custom_algorithm_step(harmony_memory, best_harmony):
            new_harmony_memory = []
            for harmony in harmony_memory:
                new_harmony = np.zeros_like(harmony)
                for i in range(len(harmony)):
                    if np.random.rand() < self.hmcr:
                        if np.random.rand() < self.par:
                            new_harmony[i] = best_harmony[i]
                        else:
                            idx = np.random.choice(self.harmony_memory_size)
                            new_harmony[i] = harmony_memory[idx][i]
                    else:
                        new_harmony[i] = np.random.uniform(-5.0, 5.0)
                    if np.random.rand() < self.mw:
                        new_harmony[i] += np.random.uniform(-1, 1)
                        if np.random.rand() < self.cr:
                            new_harmony[i] += np.random.uniform(-1, 1)
                    if np.random.rand() < 0.3:  # Probability change for line refinement
                        new_harmony[i] += np.random.uniform(-0.5, 0.5)  # Line refinement
                new_harmony_memory.append(new_harmony)
            return np.array(new_harmony_memory)

        harmony_memory = initialize_harmony_memory()
        best_harmony = harmony_memory[np.argmin([func(h) for h in harmony_memory])]
        remaining_budget = self.budget - self.harmony_memory_size

        while remaining_budget > 0:
            new_harmony_memory = custom_algorithm_step(harmony_memory, best_harmony)
            for idx, harmony in enumerate(new_harmony_memory):
                if remaining_budget <= 0:
                    break
                new_fitness = func(harmony)
                if new_fitness < func(harmony_memory[idx]):
                    harmony_memory[idx] = harmony
                    if new_fitness < func(best_harmony):
                        best_harmony = harmony
                remaining_budget -= 1

        return best_harmony

# Example usage:
# optimizer = CustomMetaheuristicAlgorithm(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function