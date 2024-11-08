import numpy as np

class EnhancedPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.lb = -5.0
        self.ub = 5.0
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.9  # optimized differential weight
        self.CR = 0.8  # optimized crossover probability

    def __call__(self, func):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        P = X.copy()
        P_fitness = np.array([func(p) for p in P])
        G = P[np.argmin(P_fitness)]
        G_fitness = np.min(P_fitness)

        evals = self.pop_size

        for iter_num in range(self.max_iter):
            if evals >= self.budget:
                break

            w = self.w_max - ((self.w_max - self.w_min) * iter_num / self.max_iter)  # adaptive inertia weight
            r1, r2 = np.random.rand(self.pop_size, 1), np.random.rand(self.pop_size, 1)
            V = w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (G - X)
            X = np.clip(X + V, self.lb, self.ub)

            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = X[idxs]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, X[i])
                
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < P_fitness[i]:
                    P[i], P_fitness[i] = trial, trial_fitness
                    if trial_fitness < G_fitness:
                        G, G_fitness = trial, trial_fitness

                if evals >= self.budget:
                    break

        return G