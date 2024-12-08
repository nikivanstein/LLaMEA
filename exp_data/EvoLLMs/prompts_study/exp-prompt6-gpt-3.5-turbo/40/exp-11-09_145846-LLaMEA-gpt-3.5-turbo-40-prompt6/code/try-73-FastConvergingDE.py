import numpy as np

class FastConvergingDE:
    def __init__(self, budget, dim, F=0.5, CR=0.9, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.adapt_rate = adapt_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        
        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant = population[a] + self.F * (population[best_index] - population[a]) + self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial

            # Adaptive mechanism to adjust mutation rates
            if np.random.rand() < self.adapt_rate:
                self.F = max(0, min(1, self.F + np.random.normal(0, 0.1)))
                self.CR = max(0, min(1, self.CR + np.random.normal(0, 0.1)))

        return best_solution