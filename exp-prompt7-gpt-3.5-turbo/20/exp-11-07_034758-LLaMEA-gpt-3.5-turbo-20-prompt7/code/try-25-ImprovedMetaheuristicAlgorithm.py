import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best_solution = population[idx[0]]
            worst_solution = population[idx[-1]]
            mutation_step = np.abs(best_solution - worst_solution) * np.random.rand(self.dim)
            population[idx[-1]] = best_solution + mutation_step
            fitness[idx[-1]] = func(population[idx[-1]])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution