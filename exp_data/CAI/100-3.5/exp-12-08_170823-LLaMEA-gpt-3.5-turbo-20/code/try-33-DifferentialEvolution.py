import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = np.random.choice(np.delete(np.arange(self.budget), j), 3, replace=False)
                a, b, c = population[idxs]
                
                rand_CR = np.clip(np.random.normal(self.CR, 0.1), 0, 1)
                rand_F = np.clip(np.random.normal(self.F, 0.1), 0, 2)
                
                mutant = np.clip(a + rand_F * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < rand_CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

        return self.f_opt, self.x_opt