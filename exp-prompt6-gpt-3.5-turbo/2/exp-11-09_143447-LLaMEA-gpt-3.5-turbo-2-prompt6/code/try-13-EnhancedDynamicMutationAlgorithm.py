import numpy as np

class EnhancedDynamicMutationAlgorithm:
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
            mutation_rates = [self.mutation_rate + 0.01 * (fitness[best_idx] - f) for f in fitness]  # Dynamic mutation adjustment based on fitness
            diversity_factor = np.std(population, axis=0)  # Measure population diversity
            mutation_rates += 0.5 * diversity_factor * np.random.randn(self.budget)  # Incorporate population diversity

            for ind, rate in zip(population, mutation_rates):
                mutated = ind + rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual