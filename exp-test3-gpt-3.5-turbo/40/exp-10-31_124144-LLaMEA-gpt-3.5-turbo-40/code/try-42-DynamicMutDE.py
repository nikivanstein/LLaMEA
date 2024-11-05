import numpy as np

class DynamicMutDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        f = 0.5 + np.random.rand() * 0.5  # Dynamic mutation factor
        for _ in range(self.budget - 1):
            idxs = np.random.choice(self.budget, 3, replace=False)
            mutant = population[idxs[0]] + f * (population[idxs[1]] - population[idxs[2]])
            crossover_prob = np.random.rand(self.dim) < 0.9
            trial = np.where(crossover_prob, mutant, population[_])
            if func(trial) < func(population[_]):
                population[_] = trial
                if func(trial) < func(best_solution):
                    best_solution = trial
        return best_solution