import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_pop_size = 20 * dim
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        pop_size = self.init_pop_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = pop_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        while evals < self.budget:
            pop_size = max(5, int(self.init_pop_size * (1 - (evals / self.budget))))  # Dynamic population size
            for i in range(pop_size):
                # Selective Mutation Strategy
                indices = np.argsort(fitness)[:pop_size//2]  # select top half based on fitness
                if i in indices:
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                else:
                    a, b, c = population[np.random.choice(pop_size, 3, replace=False)]
                
                decay = 1 - evals / self.budget
                dynamic_F = self.F * (1 - 0.5 * (evals / self.budget))
                mutant = np.clip(a + dynamic_F * decay * (b - c), self.lower_bound, self.upper_bound)
                
                # Enhanced Crossover Strategy
                adaptive_CR = 0.2 if evals > 0.8 * self.budget else self.CR  # Adjusted crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_CR
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
                
                # Local Search (Hill-Climbing) with dynamic step size
                if evals % 100 == 0:
                    direction = np.random.uniform(-1.0, 1.0, self.dim)
                    step = (0.01 + 0.99 * (evals / self.budget)) * (self.upper_bound - self.lower_bound)
                    local_trial = np.clip(best_solution + step * direction, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[best_idx]:
                        best_solution = local_trial
                        fitness[best_idx] = local_fitness
        
        return best_solution