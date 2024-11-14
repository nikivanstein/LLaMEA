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
                # Mutation with dynamic F adjustment
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                decay = 1 - evals / self.budget
                dynamic_F = self.F * (1 - 0.5 * (evals / self.budget))
                
                # Adding a small adaptive noise factor
                noise_factor = np.random.normal(0, 0.01, self.dim)
                
                mutant = np.clip(a + dynamic_F * decay * (b - c) + noise_factor, self.lower_bound, self.upper_bound)
                
                # Adaptive Crossover Probability
                self.CR = 0.1 if evals > 0.8 * self.budget else 0.9

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