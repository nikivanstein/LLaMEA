import numpy as np

class ImprovedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Initial crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size
        mutation_factor = np.ones(self.pop_size) * 0.5  # Initialize mutation factors
        crossover_prob = np.ones(self.pop_size) * self.CR  # Adaptive crossover
        
        while evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                self.F = mutation_factor[i]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.array([mutant[j] if np.random.rand() < crossover_prob[i] else pop[i][j] for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    mutation_factor[i] = min(mutation_factor[i] + 0.1, 1.0)
                    crossover_prob[i] = min(crossover_prob[i] + 0.05, 1.0)  # Adaptive crossover increase
                else:
                    mutation_factor[i] = max(mutation_factor[i] - 0.1, 0.1)
                    crossover_prob[i] = max(crossover_prob[i] - 0.05, 0.1)  # Adaptive crossover decrease
                
                if evals >= self.budget:
                    break
                
                # Gradient-based mutation
                if evals < self.budget:
                    grad_perturb = np.random.randn(self.dim) * 0.05  # Smaller perturbation
                    local_trial = pop[i] + grad_perturb
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