import numpy as np

class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_decay = 0.9
        self.CR_growth = 1.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        F, CR = self.F_max, self.CR_max

        for _ in range(self.budget):
            for i in range(len(population)):
                a, b, c = np.random.choice(len(population), 3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = np.random.choice(len(population), 3, replace=False)

                mutant = population[a] + F * (population[b] - population[c])
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

            F *= self.F_decay
            CR *= self.CR_growth

        return best_solution