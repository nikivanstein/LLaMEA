import numpy as np

class DynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        mutation_probs = np.ones(self.dim) * 0.5
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]

            for i, individual in enumerate(self.population):
                if i != best_idx:
                    fitness_difference = (fitness[best_idx] - fitness[i]) / fitness[best_idx]
                    mutation_scale = 1.0 / (1.0 + np.exp(-5 * fitness_difference))
                    mutant = individual + np.random.normal(0, mutation_probs * mutation_scale)
                    self.population[i] = mutant

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual