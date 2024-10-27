import numpy as np

class HybridFireflyLevyOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            
            # Firefly movement based on attraction and Levy flight for exploration
            for i in range(self.budget):
                attractiveness = 1 / (1 + np.linalg.norm(self.population[i] - best_solution))
                step_size = 0.1 * np.random.standard_cauchy(size=self.dim)
                new_position = self.population[i] + attractiveness * step_size
                new_position = np.clip(new_position, -5.0, 5.0)
                if func(new_position) < fitness[i]:
                    self.population[i] = new_position
        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]
        
        return best_solution