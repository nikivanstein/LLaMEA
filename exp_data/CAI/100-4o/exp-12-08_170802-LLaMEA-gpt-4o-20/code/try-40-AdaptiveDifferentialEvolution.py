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
        self.dynamic_pop_size = pop_size  # Initialize dynamic population size

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.dynamic_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.dynamic_pop_size
        
        while evals < self.budget:
            for i in range(self.dynamic_pop_size):
                # Mutation
                indices = list(range(self.dynamic_pop_size))
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
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                # Self-adaptive mechanism
                if evals % (self.dynamic_pop_size * 8) == 0:  # Adjust self-adaptive frequency
                    f_mean = fitness.mean()
                    if self.f_opt < f_mean:
                        self.F = min(self.F + 0.1, 1.0)
                        self.CR = max(self.CR - 0.1, 0.1)
                        self.dynamic_pop_size = min(self.dynamic_pop_size + 2, int(self.pop_size * 1.5))  # Increase pop size
                    else:
                        self.F = max(self.F - 0.1, 0.1)
                        self.CR = min(self.CR + 0.1, 1.0)
                        self.dynamic_pop_size = max(self.dynamic_pop_size - 2, int(self.pop_size * 0.5))  # Decrease pop size

        return self.f_opt, self.x_opt