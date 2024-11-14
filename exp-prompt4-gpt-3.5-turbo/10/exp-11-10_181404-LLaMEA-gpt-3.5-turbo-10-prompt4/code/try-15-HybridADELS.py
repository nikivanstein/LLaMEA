import numpy as np

class HybridADELS:
    def __init__(self, budget, dim, F=0.5, CR=0.9, ls_iters=10):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.ls_iters = ls_iters

    def local_search(self, func, ind):
        current_fitness = func(ind)
        for _ in range(self.ls_iters):
            perturbed = np.clip(ind + 0.1 * np.random.randn(self.dim), -5.0, 5.0)
            perturbed_fitness = func(perturbed)
            if perturbed_fitness < current_fitness:
                ind = perturbed
                current_fitness = perturbed_fitness
        return ind

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        for _ in range(self.budget - pop_size):
            pop_size = int(10 * (1 - _ / self.budget) * self.dim)
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial = self.local_search(func, trial)
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial

        return best