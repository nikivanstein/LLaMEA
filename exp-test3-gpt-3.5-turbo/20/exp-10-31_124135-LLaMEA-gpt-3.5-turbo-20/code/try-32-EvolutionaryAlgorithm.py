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
            
            # Mutation operator based on Gaussian distribution
            mutated_population = population + np.random.normal(0, 0.1, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget
        
        return best_solution