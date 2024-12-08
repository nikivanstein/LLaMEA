import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        while evals < self.budget:
            for i in range(self.pop_size):
                # Adaptive Mutation Control
                if evals > 0.5 * self.budget:
                    self.F = 0.9 * (1 - evals / self.budget)
                
                # Selective Mutation Strategy
                indices = np.argsort(fitness)[:self.pop_size//2]
                if i in indices:
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                else:
                    a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                
                dynamic_F = self.F * (1 - 0.5 * (evals / self.budget))
                mutant = np.clip(a + dynamic_F * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                
                # Dynamic Population Resizing
                if evals % 150 == 0 and self.pop_size > 4 * self.dim:
                    self.pop_size = max(4 * self.dim, int(self.pop_size * 0.9))
                    population = population[:self.pop_size]
                    fitness = fitness[:self.pop_size]
        
        return best_solution