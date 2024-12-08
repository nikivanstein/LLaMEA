import numpy as np

class AdaptiveMutationAlgorithm:
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

            new_population = []
            diversity = np.mean(np.std(population, axis=0))  # Measure population diversity
            for ind in population:
                mutation_rate = self.mutation_rate + 0.01 * (fitness[best_idx] - func(ind)) + 0.005 * diversity  # Adaptive mutation adjustment
                mutated = ind + mutation_rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual