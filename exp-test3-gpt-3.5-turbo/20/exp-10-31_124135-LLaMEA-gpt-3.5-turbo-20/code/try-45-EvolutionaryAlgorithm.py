import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, elite_frac=0.1):
        self.budget = budget
        self.dim = dim
        self.elite_frac = elite_frac

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            # Mutation operator based on Gaussian distribution
            mutated_population = population + np.random.normal(0, 0.1, (self.budget, self.dim))
            
            # Introduce elitism by retaining the top individuals
            num_elites = int(self.elite_frac * self.budget)
            elite_idx = np.argsort(fitness)[:num_elites]
            mutated_population[elite_idx] = population[elite_idx]
            
            population = mutated_population
            evals += self.budget
        
        return best_solution