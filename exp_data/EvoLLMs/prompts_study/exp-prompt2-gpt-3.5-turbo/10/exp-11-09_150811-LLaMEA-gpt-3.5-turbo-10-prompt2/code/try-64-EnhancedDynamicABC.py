import numpy as np

class EnhancedDynamicABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_strength = 1.0
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    trial_solution = self.population[i] + np.random.uniform(-self.mutation_strength, self.mutation_strength, self.dim) * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
                        self.mutation_strength *= 0.95  # Adaptive mutation adjustment
                    else:
                        self.mutation_strength *= 1.05  # Adaptive mutation adjustment
        return best_solution