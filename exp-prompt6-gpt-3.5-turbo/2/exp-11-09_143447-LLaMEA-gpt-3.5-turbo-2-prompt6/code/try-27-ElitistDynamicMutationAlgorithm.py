import numpy as np

class ElitistDynamicMutationAlgorithm:
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

            new_population = [best_individual]  # Elitism - preserving the best individual
            for _ in range(1, self.budget):
                mutation_rate = self.mutation_rate + 0.01 * (fitness[best_idx] - func(population[_]))  # Dynamic mutation adjustment
                mutated = population[_] + mutation_rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual