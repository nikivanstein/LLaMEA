import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        for _ in range(self.budget):
            idxs = np.random.choice(self.budget, 3, replace=False)
            target, a, b = population[idxs]
            mutated = target + 0.5 * (a - b)
            crossover = np.random.rand(self.dim) < 0.9
            trial = np.where(crossover, mutated, target)
            if func(trial) < func(target):
                population[idxs[0]] = trial
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution