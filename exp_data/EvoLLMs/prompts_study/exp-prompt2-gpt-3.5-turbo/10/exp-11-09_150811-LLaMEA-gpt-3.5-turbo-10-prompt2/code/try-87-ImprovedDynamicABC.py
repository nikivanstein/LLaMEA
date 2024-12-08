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
                    # Incorporate Levy flights for enhanced exploration
                    levy = np.random.standard_cauchy(self.dim) / np.sqrt(np.abs(np.random.normal(0, 1, self.dim)))
                    trial_solution = self.population[i] + levy * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution