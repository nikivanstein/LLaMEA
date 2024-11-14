import numpy as np

class FastConvergingDynamicMutationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_mutation_rate = 0.1  # Initial mutation rate

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim)) + 0.1 * np.random.randn(self.budget, self.dim)

    def __call__(self, func):
        population = self.initialize_population()

        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            new_population = []
            for idx, ind in enumerate(population):
                mutation_rate = self.base_mutation_rate + 0.01 * (fitness[best_idx] - fitness[idx])
                mutated = ind + mutation_rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual