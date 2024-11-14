import numpy as np
import concurrent.futures

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        fitness = np.array([func(ind) for ind in population])
        
        best_solution = population[np.argmin(fitness)]
        return best_solution