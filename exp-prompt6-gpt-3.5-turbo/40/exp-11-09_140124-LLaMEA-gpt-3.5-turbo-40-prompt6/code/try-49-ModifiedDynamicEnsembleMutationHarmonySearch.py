import numpy as np

class ModifiedDynamicEnsembleMutationHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 10
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        mutation_rates = np.random.uniform(0.1, 0.3, population_size)

        for _ in range(self.budget):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            for i in range(population_size):
                mutation_rate = mutation_rates[i]
                mutation_factor = np.random.uniform(0.1, 0.5, self.dim)
                if np.random.rand() < mutation_rate:
                    if np.random.rand() < 0.5:
                        new_solution = harmony_memory[i] + mutation_factor * (new_solution - harmony_memory[i])
                    else:
                        new_solution = harmony_memory[i] - mutation_factor * (new_solution - harmony_memory[i])

                if func(new_solution) < func(harmony_memory[i]):
                    harmony_memory[i] = new_solution
                    mutation_rates[i] += 0.05 * (1 - func(new_solution) / func(harmony_memory[i]))  # Update mutation rate based on individual fitness

        best_solution = min(harmony_memory, key=func)
        return best_solution