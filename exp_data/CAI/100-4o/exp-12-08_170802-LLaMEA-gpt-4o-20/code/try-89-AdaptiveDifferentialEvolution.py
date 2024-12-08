import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.pop_size
        
        while evals < self.budget:
            # Dynamic Population Size Adjustment
            self.pop_size = max(5, int(self.pop_size - (evals / self.budget) * (self.pop_size - 5)))
            new_population = np.empty((self.pop_size, self.dim))
            new_fitness = np.empty(self.pop_size)
            
            for i in range(self.pop_size):
                # Mutation
                indices = list(range(len(population)))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    
            population = new_population
            fitness = new_fitness
            
            # Self-adaptive mechanism
            if evals % (self.pop_size * 10) == 0:
                f_mean = fitness.mean()
                if self.f_opt < f_mean:
                    self.F = min(self.F + 0.1, 1.0)
                    self.CR = max(self.CR - 0.1, 0.1)
                else:
                    self.F = max(self.F - 0.1, 0.1)
                    self.CR = min(self.CR + 0.1, 1.0)

        return self.f_opt, self.x_opt