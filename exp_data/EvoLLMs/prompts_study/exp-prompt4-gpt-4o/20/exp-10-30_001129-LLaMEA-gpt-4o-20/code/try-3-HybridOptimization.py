import numpy as np

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size
        adaptive_step_size = 0.1
        
        while evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.array([mutant[j] if np.random.rand() < self.CR else pop[i][j] for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    adaptive_step_size = max(0.01, adaptive_step_size * 0.9)  # Decrease step size

                if evals < self.budget:
                    local_trial = pop[i] + np.random.normal(0, adaptive_step_size, self.dim)
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    
                    if local_fitness < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = local_fitness
                        adaptive_step_size = max(0.01, adaptive_step_size * 0.9)  # Decrease step size
                    else:
                        adaptive_step_size = min(0.2, adaptive_step_size * 1.1)  # Increase step size
            
            if evals >= self.budget:
                break
        
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]