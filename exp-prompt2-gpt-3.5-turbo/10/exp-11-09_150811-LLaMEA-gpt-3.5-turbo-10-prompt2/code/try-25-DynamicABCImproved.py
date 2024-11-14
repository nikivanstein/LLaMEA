import numpy as np

class DynamicABCImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(len(self.population)):
                if i != best_idx:
                    trial_solution = self.population[i] + np.random.uniform(-1, 1, self.dim) * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
                        
            if np.random.rand() < 0.1:  # 10% chance to add/remove a random individual
                if len(self.population) < self.budget and np.random.rand() < 0.5:  # Add a new individual
                    self.population = np.append(self.population, [np.random.uniform(-5.0, 5.0, self.dim)], axis=0)
                elif len(self.population) > 1:  # Remove a random individual
                    del_idx = np.random.choice(np.arange(len(self.population)))
                    self.population = np.delete(self.population, del_idx, axis=0)
        return best_solution