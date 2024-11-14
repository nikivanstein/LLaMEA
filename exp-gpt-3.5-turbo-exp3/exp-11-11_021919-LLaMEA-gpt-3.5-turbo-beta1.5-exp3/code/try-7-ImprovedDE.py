import numpy as np

class ImprovedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_min, self.F_max = 0.1, 0.9
        self.CR_min, self.CR_max = 0.1, 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        F, CR = self.F_max, self.CR_max

        for _ in range(self.budget):
            for i in range(len(population)):
                idxs = np.arange(len(population))
                idxs = idxs[idxs != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)

                beta = np.random.uniform(0, 1)
                mutant = population[a] + beta * (population[b] - population[c])
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