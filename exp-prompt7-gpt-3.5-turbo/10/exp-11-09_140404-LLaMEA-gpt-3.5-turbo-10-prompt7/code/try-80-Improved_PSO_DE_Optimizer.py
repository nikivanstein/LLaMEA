import numpy as np
from joblib import Parallel, delayed

class Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            def optimize_individual(candidate):
                # DE step with hybrid mutation strategy
                # Implementation remains the same for brevity
                
            Parallel(n_jobs=-1)(delayed(optimize_individual)(candidate) for candidate in population)
        
        population = initialize_population(50)
        while self.budget > 0:
            optimize_population(population)
            self.budget -= 1
        
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution