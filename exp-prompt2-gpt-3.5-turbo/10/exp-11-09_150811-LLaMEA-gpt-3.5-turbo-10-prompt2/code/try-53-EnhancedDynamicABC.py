import numpy as np

class EnhancedDynamicABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    exploration = np.random.uniform(-1, 1, self.dim) * (best_solution - self.population[i])
                    exploitation = np.random.uniform(-1, 1, self.dim) * (self.population[np.random.randint(self.budget)] - self.population[i])
                    trial_solution = self.population[i] + self.mutation_rate * (exploration + exploitation)
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution