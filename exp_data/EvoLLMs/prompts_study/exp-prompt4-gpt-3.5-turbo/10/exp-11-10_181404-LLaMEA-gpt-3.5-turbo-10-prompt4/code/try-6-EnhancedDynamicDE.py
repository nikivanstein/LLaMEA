import numpy as np

class EnhancedDynamicDE:
    def __init__(self, budget, dim, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        for _ in range(self.budget - pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F = np.clip(self.F + np.random.normal(0, 0.1), self.F_min, self.F_max)
                CR = np.clip(self.CR + np.random.normal(0, 0.1), self.CR_min, self.CR_max)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < CR
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