import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            # Crowding distance-based selection
            distances = np.linalg.norm(population - population[:, np.newaxis], axis=2)
            sorted_indices = np.argsort(distances, axis=1)
            selected_indices = sorted_indices[:, 1]
            population = population[selected_indices]
            mutated_population = population + np.random.normal(0, 0.1, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget

        return best_solution