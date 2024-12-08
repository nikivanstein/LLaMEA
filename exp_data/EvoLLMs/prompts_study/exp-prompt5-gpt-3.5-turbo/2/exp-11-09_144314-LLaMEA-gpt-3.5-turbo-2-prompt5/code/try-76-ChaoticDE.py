import numpy as np

class ChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                chaotic_map = np.sin(self.population[a]) * np.cos(self.population[b]) / (np.tanh(self.population[c]) + 1)
                mutant = self.population[a] + 0.5 * (self.population[b] - self.population[c]) + chaotic_map
                crossover = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover, mutant, self.population[i])
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
        best_solution = self.population[np.argmin([func(x) for x in self.population])]
        return best_solution