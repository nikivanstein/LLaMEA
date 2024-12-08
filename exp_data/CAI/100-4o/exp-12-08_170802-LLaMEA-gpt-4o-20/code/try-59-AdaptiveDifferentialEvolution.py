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
        success_rate = 0.2
        
        while evals < self.budget:
            new_population = []
            for i in range(self.pop_size):
                # Mutation
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.F * (b - c)
                # Adjust mutation strategy dynamically
                if np.random.rand() < success_rate:
                    mutant += np.random.normal(0, 0.1, size=self.dim)
                mutant = np.clip(mutant, bounds[0], bounds[1])
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population.append(population[i])
                
                # Self-adaptive mechanism
                if evals % (self.pop_size * 10) == 0:
                    success_rate = sum(f < f_m for f, f_m in zip(fitness, np.full(self.pop_size, fitness.mean()))) / self.pop_size
                    if self.f_opt < fitness.mean():
                        self.F = min(self.F + 0.1, 1.0)
                        self.CR = max(self.CR - 0.1, 0.1)
                        self.pop_size = min(self.pop_size + 1, 50)
                    else:
                        self.F = max(self.F - 0.1, 0.1)
                        self.CR = min(self.CR + 0.1, 1.0)
                        self.pop_size = max(self.pop_size - 1, 10)
            
            population = np.array(new_population)

        return self.f_opt, self.x_opt