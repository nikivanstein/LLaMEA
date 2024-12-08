import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = np.random.uniform(0.1, 0.9)  # Dynamic adaptation of F within [0.1, 0.9]
        self.CR = np.random.uniform(0.1, 0.9) # Dynamic adaptation of CR within [0.1, 0.9]
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]
                    
                self.F = max(0.1, min(0.9, self.F + 0.01))  # Update F within [0.1, 0.9]
                self.CR = max(0.1, min(0.9, self.CR + 0.01)) # Update CR within [0.1, 0.9]

        return self.f_opt, self.x_opt