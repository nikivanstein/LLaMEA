import numpy as np

class DynamicScalingDE:
    def __init__(self, budget, dim, F=0.5, CR=0.9, scaling_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.scaling_factor = scaling_factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                scaling = 1.0 / np.sqrt(1 + np.exp(-self.scaling_factor * (fitness[a] - fitness[b])))
                mutant = population[a] + scaling * self.F * (population[best_index] - population[a]) + scaling * self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial

        return best_solution