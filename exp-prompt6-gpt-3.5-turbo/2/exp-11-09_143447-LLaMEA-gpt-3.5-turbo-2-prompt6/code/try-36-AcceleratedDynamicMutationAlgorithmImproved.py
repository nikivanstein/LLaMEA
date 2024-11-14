import numpy as np

class AcceleratedDynamicMutationAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim)) + 0.1 * np.random.randn(self.budget, self.dim)

    def __call__(self, func):
        population = self.initialize_population()
        mutation_rates = np.ones(self.budget) * 0.1

        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            new_population = []
            for idx, ind in enumerate(population):
                mutation_rate = mutation_rates[idx] + 0.01 * (fitness[best_idx] - func(ind))
                mutated = ind + mutation_rate * np.random.randn(self.dim)
                mutation_rates[idx] = mutation_rate
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual