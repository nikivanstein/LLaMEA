import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_min, self.F_max = 0.2, 0.8
        self.CR_min, self.CR_max = 0.2, 0.8

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        F, CR = self.F_max, self.CR_max

        for _ in range(self.budget):
            for i in range(len(population)):
                candidates = np.random.choice(len(population), 5, replace=False)
                a, b, c, d, e = candidates

                while a == i or b == i or c == i or d == i or e == i:
                    candidates = np.random.choice(len(population), 5, replace=False)
                    a, b, c, d, e = candidates

                mutant = population[a] + F * (population[b] - population[c]) + F * (population[d] - population[e])
                mutant = np.clip(mutant, -5.0, 5.0)

                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial[j_rand] = mutant[j_rand]

                ft = func(trial)
                if ft < fitness[i]:
                    fitness[i] = ft
                    population[i] = trial

                    if ft < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = ft

            F = self.F_min + (_ / self.budget) * (self.F_max - self.F_min)
            CR = self.CR_max - (_ / self.budget) * (self.CR_max - self.CR_min)

        return best_solution