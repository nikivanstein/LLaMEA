import numpy as np

class PopulationBasedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        harmony_memory_size = 10
        harmony_memory = np.random.uniform(lower_bound, upper_bound, size=(harmony_memory_size, self.dim))
        harmony_fitness = np.array([func(sol) for sol in harmony_memory])
        for _ in range(self.budget):
            new_solution = np.mean(harmony_memory, axis=0)  # Harmonize solutions
            new_solution += np.random.uniform(-0.1, 0.1, size=self.dim)  # Introduce randomness
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = func(new_solution)
            worst_idx = np.argmax(harmony_fitness)
            if new_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_solution
                harmony_fitness[worst_idx] = new_fitness
        return harmony_memory[np.argmin(harmony_fitness)]