import numpy as np

class HarmonySearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_mutation_rate = 0.2
        self.func_evals = 0

    def __call__(self, func):
        population_size = 10
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))

        for _ in range(self.budget):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            self.func_evals += 1
            current_mutation_rate = self.initial_mutation_rate * (1 - self.func_evals / self.budget)  # Adaptive mutation rate

            if np.random.rand() < current_mutation_rate:
                mutation_factor = np.random.uniform(0.1, 0.5, self.dim)
                new_solution = harmony_memory[np.random.randint(population_size)] + mutation_factor * (new_solution - harmony_memory[np.random.randint(population_size)])

            index = np.random.randint(population_size)
            if func(new_solution) < func(harmony_memory[index]):
                harmony_memory[index] = new_solution

        best_solution = min(harmony_memory, key=func)
        return best_solution