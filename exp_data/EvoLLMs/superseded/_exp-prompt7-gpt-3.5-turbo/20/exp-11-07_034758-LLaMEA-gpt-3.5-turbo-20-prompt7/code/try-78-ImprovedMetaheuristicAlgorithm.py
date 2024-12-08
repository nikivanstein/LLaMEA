import numpy as np
import concurrent.futures

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        population_fitness = np.array([func(ind) for ind in population])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            fitness = np.array(list(executor.map(func, population)))
            best_idx = np.argmin(fitness)

        best_solution = population[best_idx]
        return best_solution