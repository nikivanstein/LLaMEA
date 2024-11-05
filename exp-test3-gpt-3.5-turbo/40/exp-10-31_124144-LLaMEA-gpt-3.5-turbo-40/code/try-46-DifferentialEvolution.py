import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        for _ in range(self.budget - 1):
            idx = np.random.choice(self.budget, 3, replace=False)
            a, b, c = population[idx]
            mutant = a + 0.8 * (b - c)
            crossover = np.random.rand(self.dim) < 0.9
            child = np.where(crossover, mutant, population[_])
            child_fitness = func(child)
            if child_fitness < fitness[_]:
                population[_] = child
                fitness[_] = child_fitness
            if child_fitness < fitness[best_idx]:
                best_solution = child
                best_idx = _
        return best_solution