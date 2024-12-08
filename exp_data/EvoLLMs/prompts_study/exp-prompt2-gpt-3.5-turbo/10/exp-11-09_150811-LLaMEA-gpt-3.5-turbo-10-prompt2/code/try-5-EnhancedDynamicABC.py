import numpy as np

class EnhancedDynamicABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_strength = np.random.uniform(0.1, 1.0, (budget, dim))  # Initialize dynamic mutation strength
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    mutation_factor = np.random.uniform(0, 1, self.dim) * self.mutation_strength[i]  # Dynamic mutation strength
                    trial_solution = self.population[i] + mutation_factor * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
                        self.mutation_strength[i] *= 1.1  # Adjust mutation strength based on success
                    else:
                        self.mutation_strength[i] *= 0.9  # Reduce mutation strength
        return best_solution