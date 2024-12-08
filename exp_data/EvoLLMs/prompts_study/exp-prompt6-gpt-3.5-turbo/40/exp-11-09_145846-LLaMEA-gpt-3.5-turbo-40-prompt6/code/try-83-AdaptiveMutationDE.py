import numpy as np

class AdaptiveMutationDE:
    def __init__(self, budget, dim, F_min=0.2, F_max=0.8, CR_min=0.5, CR_max=1.0):
        self.budget = budget
        self.dim = dim
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        
        F = np.random.uniform(self.F_min, self.F_max, self.budget)
        CR = np.random.uniform(self.CR_min, self.CR_max, self.budget)

        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant = population[a] + F[i] * (population[best_index] - population[a]) + F[i] * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR[i]
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial

        return best_solution