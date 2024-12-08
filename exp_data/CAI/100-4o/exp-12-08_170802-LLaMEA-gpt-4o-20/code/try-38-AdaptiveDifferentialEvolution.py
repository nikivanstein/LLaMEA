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
            for i in range(self.pop_size):
                # Mutation
                indices = list(range(self.pop_size))
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
                if evals % (self.pop_size * 10) == 0:
                    f_mean = fitness.mean()
                    diversity = np.std(fitness)
                    if self.f_opt < f_mean:
                        self.F = min(self.F + 0.1 * diversity, 1.0)  # Adjust F based on diversity
                        self.CR = max(self.CR - 0.05, 0.1)  # Slightly reduce CR
                        self.pop_size = min(int(self.pop_size * 1.1), 100)  # Increase population size
                    else:
                        self.F = max(self.F - 0.1, 0.1)
                        self.CR = min(self.CR + 0.05, 1.0)  # Slightly increase CR
                        self.pop_size = max(int(self.pop_size * 0.9), 5)  # Decrease population size
                    population = np.vstack((population, np.random.uniform(bounds[0], bounds[1], (self.pop_size - len(population), self.dim))))
                    fitness = np.append(fitness, np.array([func(ind) for ind in population[len(fitness):]]))
                    evals += self.pop_size - len(fitness)

        return self.f_opt, self.x_opt