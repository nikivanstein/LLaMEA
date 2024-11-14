import numpy as np

class ImprovedDynamicABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    if np.random.rand() > 0.1:  # 10% chance for Levy flight
                        beta = 1.5
                        alpha = 0.01 * np.random.standard_cauchy(self.dim)
                        trial_solution = self.population[i] + beta * alpha * (best_solution - self.population[i])
                    else:
                        trial_solution = self.population[i] + np.random.uniform(-1, 1, self.dim) * (best_solution - self.population[i])
                        
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution