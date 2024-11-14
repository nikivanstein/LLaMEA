import numpy as np

class HarmonySearchAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 10
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        mutation_rate = 0.2

        for _ in range(self.budget):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            
            # Adaptive Dynamic Mutation Strategy based on individual fitness
            mutation_rate = 0.2 + 0.1 * np.exp(-0.1 * _) + 0.2 * (func(harmony_memory.min(axis=0)) - func(harmony_memory.max(axis=0)))  # Adjust mutation rate based on individual fitness
            if np.random.rand() < mutation_rate:
                mutation_factor = np.random.uniform(0.1, 0.5, self.dim)
                new_solution = harmony_memory[np.random.randint(population_size)] + mutation_factor * (new_solution - harmony_memory[np.random.randint(population_size)])

            index = np.random.randint(population_size)
            if func(new_solution) < func(harmony_memory[index]):
                harmony_memory[index] = new_solution

        best_solution = min(harmony_memory, key=func)
        return best_solution