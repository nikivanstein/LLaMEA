import numpy as np

class AdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(20 * dim, 10)
        self.F = 0.8
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
                indices = np.argsort(fitness)[:self.pop_size//2]
                if i in indices:
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                else:
                    a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                
                decay = 1 - evals / self.budget
                self.F = 0.5 + 0.3 * np.sin(np.pi * evals / self.budget)
                mutant = np.clip(a + self.F * decay * (b - c), self.lower_bound, self.upper_bound)
                
                # Dynamic CR adjustment based on evaluation progress
                self.CR = 0.9 - 0.8 * (evals / self.budget)
                crossover_mask = np.random.rand(self.dim) < self.CR
                crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                
                if evals % 80 == 0:
                    direction = np.random.uniform(-1.0, 1.0, self.dim)
                    step_size = (0.02 + 0.98 * (evals / self.budget)) * (self.upper_bound - self.lower_bound)
                    local_trial = np.clip(best_solution + step_size * direction, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[best_idx]:
                        best_solution = local_trial
                        fitness[best_idx] = local_fitness
        
        return best_solution