import numpy as np

class MultiObjectiveDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            
            # Novel multi-objective mutation strategies
            for i in range(self.budget):
                rand_indices = np.random.choice(self.budget, 3, replace=False)
                mutant = self.population[rand_indices[0]] + 0.8 * (self.population[rand_indices[1]] - self.population[rand_indices[2]])
                crossover = np.random.rand(self.dim) < 0.9
                self.population[i] = np.where(crossover, mutant, self.population[i])
        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]
        
        return best_solution