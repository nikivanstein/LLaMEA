import numpy as np

class DynamicAdaptationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.9
        self.F = 0.5
        self.CR_adapt = np.random.uniform(0.1, 0.9, self.budget)
        self.F_adapt = np.full(self.budget, 0.5)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            mutant = a + self.F_adapt[i] * (b - c)
            crossover = np.random.rand(self.dim) < self.CR_adapt[i]
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                self.CR_adapt[i] = np.clip(self.CR_adapt[i] + 0.1 * (1 - int(trial_fitness < fitness[i])), 0.1, 0.9)
                self.F_adapt[i] = np.clip(self.F_adapt[i] + 0.1 * (1 - int(trial_fitness < fitness[i])), 0.1, 0.9)
                
        best_idx = np.argmin(fitness)
        return population[best_idx]