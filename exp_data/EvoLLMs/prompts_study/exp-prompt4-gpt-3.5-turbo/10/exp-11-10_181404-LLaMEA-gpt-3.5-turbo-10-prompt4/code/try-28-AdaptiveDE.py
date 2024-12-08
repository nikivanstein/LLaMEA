import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.CR = CR

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        F = np.full(pop_size, 0.5)

        for _ in range(self.budget - pop_size):
            pop_size = int(10 * (1 - _ / self.budget) * self.dim)
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_i = np.clip(F[i] + 0.1 * np.random.randn(), 0.1, 0.9)  # Self-adaptive mutation factor F
                mutant = np.clip(a + F_i * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    F[i] = F_i
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = trial

            if _ % 10 == 0:  # Dynamic adjustment of population size
                mean_fitness = np.mean(fitness)
                std_fitness = np.std(fitness)
                if std_fitness < 1e-6:
                    break
                pop_size = int(10 * (1 - _ / self.budget) * self.dim) if mean_fitness < best_fitness else int(10 * (1 - _ / self.budget) * self.dim * 1.1)
                best_fitness = mean_fitness

        return best