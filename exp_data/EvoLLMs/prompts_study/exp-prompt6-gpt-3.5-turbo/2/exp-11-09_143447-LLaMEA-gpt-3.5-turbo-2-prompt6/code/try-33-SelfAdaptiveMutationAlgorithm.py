import numpy as np

class SelfAdaptiveMutationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1  # Initial mutation rate

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim)) + 0.1 * np.random.randn(self.budget, self.dim)

    def diversity_measure(self, population):
        return np.mean(np.std(population, axis=0))

    def __call__(self, func):
        population = self.initialize_population()

        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            diversity = self.diversity_measure(population)

            new_population = []
            for ind in population:
                mutation_rate = self.mutation_rate + 0.01 * diversity
                mutated = ind + mutation_rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual