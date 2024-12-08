import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        F = np.full(pop_size, 0.5)
        CR = 0.9

        for _ in range(self.budget - pop_size):
            pop_size = int(10 * (1 - _ / self.budget) * self.dim)
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_i = np.clip(F[i] + 0.1 * np.random.randn(), 0.1, 0.9)  # Self-adaptive mutation factor F
                CR_i = np.clip(CR + 0.1 * np.random.randn(), 0.1, 0.9)  # Self-adaptive crossover rate CR
                mutant = np.clip(a + F_i * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < CR_i
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    F[i] = F_i
                    CR = CR_i
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial

        return best