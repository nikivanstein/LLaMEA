import numpy as np

class FastDynamicMutationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1  # Initial mutation rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            diversity = np.mean(np.std(population, axis=0))  # Calculate population diversity
            new_population = []
            for ind in population:
                mutation_rate = self.mutation_rate + 0.01 * (fitness[best_idx] - func(ind)) + 0.1 * diversity  # Dynamic mutation adjustment with diversity
                mutated = ind + mutation_rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual