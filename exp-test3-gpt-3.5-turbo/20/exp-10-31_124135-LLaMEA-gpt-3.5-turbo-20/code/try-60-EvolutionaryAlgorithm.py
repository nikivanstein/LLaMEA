import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sigma = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            # Mutation operator based on Gaussian distribution with adaptive step size
            mutated_population = population + np.random.normal(0, self.sigma, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget

            # Adapt mutation step size
            if evals % (self.budget // 10) == 0:
                self.sigma *= 0.9
        
        return best_solution