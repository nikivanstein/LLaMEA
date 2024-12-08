import numpy as np

class DynamicDE:
    def __init__(self, budget, dim, F_init=0.5, CR_init=0.9, success_threshold=0.2):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.CR_init = CR_init
        self.success_threshold = success_threshold

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        
        F = np.full(self.budget, self.F_init)
        CR = np.full(self.budget, self.CR_init)
        
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
                        F[i] *= 1.1 if f_trial < fitness[i] else 0.9
                        CR[i] *= 1.1 if f_trial < fitness[i] else 0.9

        return best_solution