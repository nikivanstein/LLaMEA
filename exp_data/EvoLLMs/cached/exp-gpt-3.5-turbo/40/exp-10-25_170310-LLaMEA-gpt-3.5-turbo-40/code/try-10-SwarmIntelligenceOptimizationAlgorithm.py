import numpy as np

class SwarmIntelligenceOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            
            # Dynamic communication topology update rule
            for i in range(self.budget):
                neighbors = np.random.choice(self.budget, int(0.5*self.budget), replace=False)
                global_best = self.population[np.argmin(fitness)]
                self.population[i] += 0.1 * np.random.uniform() * (global_best - self.population[i])
        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]
        
        return best_solution