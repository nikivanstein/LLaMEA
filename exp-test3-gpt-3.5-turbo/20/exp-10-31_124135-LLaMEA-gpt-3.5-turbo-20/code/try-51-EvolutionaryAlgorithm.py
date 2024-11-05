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
            
            mutated_population = population + np.random.normal(0, 0.1, (self.budget, self.dim))
            
            # Introducing crossover operation
            crossover_point = np.random.randint(0, self.dim, self.budget//2)
            for i in range(0, self.budget, 2):
                population[i, crossover_point[i//2]:] = mutated_population[i+1, crossover_point[i//2]:]
                population[i+1, crossover_point[i//2]:] = mutated_population[i, crossover_point[i//2]:]
            
            evals += self.budget
        
        return best_solution