import numpy as np

class HybridOptimizationImproved:
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
        
        best_idx = np.argmin(fitness)  # Track the index of the best solution
        best_solution = pop[best_idx].copy()  # Track the best solution
        
        while evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                self.F = 0.5 + np.random.rand() * 0.5  # Adaptive mutation
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.array([mutant[j] if np.random.rand() < self.CR else pop[i][j] for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                # Local search: Adaptive Random Search
                if evals < self.budget:
                    perturbation_size = 0.1 * (1.0 - evals / self.budget)
                    local_trial = pop[i] + np.random.uniform(-perturbation_size, perturbation_size, self.dim)
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    
                    if local_fitness < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = local_fitness
            
            # Elitism: Retain the best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < fitness[best_idx]:
                best_solution = pop[current_best_idx].copy()
                best_idx = current_best_idx
            
            if evals >= self.budget:
                break
        
        return best_solution, fitness[best_idx]