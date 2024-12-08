import numpy as np

class DynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_probs = np.ones(dim) * 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]

            new_population = []
            for idx, individual in enumerate(self.population):
                performance_ratio = fitness[idx] / min(fitness)
                mutation_strength = 0.1 + 0.4 * performance_ratio
                mutant = individual + np.random.normal(0, mutation_strength, self.dim)
                new_population.append(mutant)

            self.population = np.array(new_population)

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual