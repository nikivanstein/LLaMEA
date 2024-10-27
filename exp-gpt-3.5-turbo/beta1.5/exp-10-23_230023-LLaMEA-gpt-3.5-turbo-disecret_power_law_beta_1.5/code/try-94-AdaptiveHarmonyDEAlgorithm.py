import numpy as np

class AdaptiveHarmonyDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.hmcr = 0.8
        self.par = 0.5
        self.F = 0.5
        self.CR = 0.3
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x, pop):
        idxs = np.random.choice(len(pop), 3, replace=False)
        a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
        return np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound) if np.random.rand() < self.CR else x

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        pbest = population.copy()
        gbest = pbest[np.argmin([func(p) for p in pbest])

        for _ in range(self.budget - self.pop_size):
            new_pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        new_pop[i, j] = pbest[np.random.randint(self.pop_size), j]
                    else:
                        new_pop[i, j] = np.random.uniform(self.lower_bound, self.upper_bound)
                if np.random.rand() < self.par:
                    new_pop[i] = self.mutate(new_pop[i], new_pop)

            pbest = np.array([x if func(x) < func(y) else y for x, y in zip(new_pop, pbest)])
            gbest = pbest[np.argmin([func(p) for p in pbest])]

        return gbest