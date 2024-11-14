import numpy as np

class EnhancedDynamicABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.history = np.zeros(budget)
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    scaling_factor = 1 / (1 + self.history[i])
                    mutation_step = scaling_factor * np.random.uniform(-1, 1, self.dim)
                    trial_solution = self.population[i] + mutation_step * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
                        self.history[i] += 1
        return best_solution