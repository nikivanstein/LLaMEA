import numpy as np

class HybridDEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.F = 0.9  # Slightly increased to enhance exploration
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
                # Mutation
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Adaptive Crossover Probability
                self.CR = 0.1 if evals > 0.75 * self.budget else 0.9  # Changed threshold for CR adaptation

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
                
                # Enhanced Local Search (Dynamic Step Size) 
                if evals % 80 == 0:  # Adjusted frequency of local search
                    direction = np.random.uniform(-1.0, 1.0, self.dim)
                    step = 0.02 * (self.upper_bound - self.lower_bound)  # Adjusted step size
                    local_trial = np.clip(best_solution + step * direction, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[best_idx]:
                        best_solution = local_trial
                        fitness[best_idx] = local_fitness
        
        return best_solution