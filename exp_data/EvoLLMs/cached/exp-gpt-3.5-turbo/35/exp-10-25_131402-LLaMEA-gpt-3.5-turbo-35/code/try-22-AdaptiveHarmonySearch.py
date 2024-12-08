import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hmcr = 0.85  # Harmony Memory Consideration Rate
        self.par = 0.4  # Pitch Adjustment Rate
        self.bandwidth = 0.01  # Bandwidth for pitch adjustment

    def __call__(self, func):
        harmony_memory_size = 10 * self.dim
        lower_bound = -5.0 * np.ones(self.dim)
        upper_bound = 5.0 * np.ones(self.dim)
        harmony_memory = np.random.uniform(low=lower_bound, high=upper_bound, size=(harmony_memory_size, self.dim))

        for _ in range(self.budget - harmony_memory_size):
            new_solution = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[i] = np.random.choice(harmony_memory[:, i])
                else:
                    new_solution[i] = np.random.uniform(lower_bound[i], upper_bound[i])

                if np.random.rand() < self.par:
                    new_solution[i] += np.random.uniform(-self.bandwidth, self.bandwidth)

                new_fitness = func(new_solution)
                if new_fitness < func(harmony_memory[-1]):
                    harmony_memory[-1] = new_solution

        best_idx = np.argmin([func(individual) for individual in harmony_memory])
        best_solution = harmony_memory[best_idx]
        best_fitness = func(best_solution)

        return best_solution, best_fitness