import numpy as np

class SelfAdaptiveMutationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.9
        self.F = 0.5
        self.success_rates = np.zeros(budget)
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            mutant = a + self.F * (b - c)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                self.success_rates[i] += 1
            else:
                self.success_rates[i] -= 1
            
            if self.success_rates[i] > 0:  # Increase mutation parameters
                self.F = min(self.F * 1.2, 0.9)
                self.CR = min(self.CR + 0.05, 0.9)
            else:  # Decrease mutation parameters
                self.F = max(self.F / 1.2, 0.1)
                self.CR = max(self.CR - 0.05, 0.1)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]