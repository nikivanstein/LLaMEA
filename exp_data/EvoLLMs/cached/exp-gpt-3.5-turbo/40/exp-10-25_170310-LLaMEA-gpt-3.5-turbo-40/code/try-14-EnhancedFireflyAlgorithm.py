import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            
            # Enhanced dynamic attraction update rule inspired by firefly algorithm
            alpha = 0.5  # Attraction coefficient
            beta_min = 0.2  # Minimum beta value
            beta_max = 1.0  # Maximum beta value
            beta = beta_min + (_ / self.budget) * (beta_max - beta_min)
            for i in range(self.budget):
                distance = np.linalg.norm(self.population[i] - best_solution)
                attractiveness = alpha * np.exp(-beta * distance)
                self.population[i] += attractiveness * np.random.uniform(-1, 1, self.dim)
        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]
        
        return best_solution