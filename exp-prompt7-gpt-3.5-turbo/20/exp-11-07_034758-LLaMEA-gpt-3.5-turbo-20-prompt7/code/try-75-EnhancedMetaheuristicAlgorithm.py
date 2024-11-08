import numpy as np
import concurrent.futures

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        def evaluate_fitness(ind):
            return func(ind)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            fitness = np.array([evaluate_fitness(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution