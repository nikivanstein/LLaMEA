import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.lb = -5.0
        self.ub = 5.0
        self.w = 0.5  # inertia weight
        self.c1 = 1.5 # cognitive (particle) weight
        self.c2 = 1.5 # social (swarm) weight
        self.F = 0.8  # differential weight
        self.CR = 0.9 # crossover probability

    def __call__(self, func):
        # Initialize particles
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        P = X.copy()  # personal best positions
        P_fitness = np.array([func(p) for p in P])
        G = P[np.argmin(P_fitness)]  # global best position
        G_fitness = np.min(P_fitness)

        evals = self.pop_size

        for _ in range(self.max_iter):
            if evals >= self.budget:
                break

            # Particle Swarm Optimization step
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            V = self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (G - X)
            X = X + V
            X = np.clip(X, self.lb, self.ub)

            # Differential Evolution step
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = X[np.random.choice(indices, 3, replace=False)]
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