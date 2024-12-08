import numpy as np

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size
        
        while evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                self.F = 0.5 + np.random.rand() * 0.5
                mutant = np.clip(a + self.F * (b - c) + self.levy_flight(self.dim), self.lower_bound, self.upper_bound)
                
                trial = np.array([mutant[j] if np.random.rand() < self.CR else pop[i][j] for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                if evals < self.budget:
                    perturbation_size = 0.1 * (1.0 - evals / self.budget)
                    local_trial = pop[i] + np.random.uniform(-perturbation_size, perturbation_size, self.dim) * (np.random.rand(self.dim) < 0.5)
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    
                    if local_fitness < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = local_fitness
            
            if evals >= self.budget:
                break
        
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]