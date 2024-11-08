import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.cr = 0.5
        self.f = 0.5
        self.w = 0.5
        self.c1 = 2.0
        self.c2 = 2.0

    def de_pbest(self, pop, pbest, func):
        mutant_pop = pop + self.f * (pbest - pop)
        cross_points = np.random.rand(*mutant_pop.shape) < self.cr
        cross_points[0, np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant_pop, pop)

    def optimize(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        pbest = pop[np.argmin([func(ind) for ind in pop])]
        gbest = pbest.copy()

        for _ in range(self.budget):
            v = self.w * v + self.c1 * np.random.rand() * (pbest - pop) + self.c2 * np.random.rand() * (gbest - pop)
            pop = self.de_pbest(pop, pbest, func)
            pbest = pop[np.argmin([func(ind) for ind in pop])]
            gbest = pbest if func(pbest) < func(gbest) else gbest

        return gbest