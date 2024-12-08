import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        f_history = [fitness[best_index]]
        F_history = [self.F]
        CR_history = [self.CR]
        
        for _ in range(self.budget):
            F_current = np.random.normal(self.F, 0.1)
            CR_current = np.random.normal(self.CR, 0.1)
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant = population[a] + F_current * (population[best_index] - population[a]) + F_current * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR_current
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial
            f_history.append(fitness[best_index])
            F_history.append(F_current)
            CR_history.append(CR_current)
        
        return best_solution