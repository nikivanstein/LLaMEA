import numpy as np

class AdaptiveFastConvergingDE:
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

        F_history = [self.F] * self.budget
        CR_history = [self.CR] * self.budget

        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)

                if np.random.rand() < self.adapt_rate:
                    self.F = np.clip(np.random.normal(0.5, 0.1), 0, 1)
                    self.CR = np.clip(np.random.normal(0.9, 0.1), 0, 1)

                mutant = population[a] + F_history[i] * (population[best_index] - population[a]) + F_history[i] * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR_history[i]
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial
                        F_history[i] = self.F
                        CR_history[i] = self.CR

        return best_solution