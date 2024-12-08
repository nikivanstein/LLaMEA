import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evaluations = 0
        while evaluations < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_ind = population[best_idx]
            mean_ind = np.mean(population, axis=0)
            new_population = 0.5 * population + 0.5 * mean_ind
            new_population[best_idx] = best_ind
            population = new_population
            evaluations += self.budget
        return best_ind