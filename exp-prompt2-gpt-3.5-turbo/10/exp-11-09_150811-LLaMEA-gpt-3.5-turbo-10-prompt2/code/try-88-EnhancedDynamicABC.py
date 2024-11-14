import numpy as np

class EnhancedDynamicABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def levy_flight(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        s = np.random.normal(0, sigma, self.dim)
        
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1 / beta)
        
        return s * step

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    levy_step = self.levy_flight()
                    trial_solution = self.population[i] + levy_step * (best_solution - self.population[i])
                    
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution