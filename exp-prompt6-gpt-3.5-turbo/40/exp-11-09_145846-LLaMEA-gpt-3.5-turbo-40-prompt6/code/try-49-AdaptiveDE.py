import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim, F=0.5, CR=0.9, F_lb=0.1, F_ub=0.9, CR_lb=0.1, CR_ub=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.CR_lb = CR_lb
        self.CR_ub = CR_ub

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                F_val = np.random.uniform(self.F_lb, self.F_ub)
                CR_val = np.random.uniform(self.CR_lb, self.CR_ub)
                mutant = population[a] + F_val * (population[best_index] - population[a]) + F_val * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR_val
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial

        return best_solution