import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        for _ in range(self.budget - pop_size):
            pop_size = int(10 * (1 - _ / self.budget) * self.dim)
            self.F = np.clip(np.random.normal(self.F, 0.1), 0, 2)  # Self-adaptive mutation factor
            self.CR = np.clip(np.random.normal(self.CR, 0.1), 0, 1)  # Self-adaptive crossover rate
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial

        return best