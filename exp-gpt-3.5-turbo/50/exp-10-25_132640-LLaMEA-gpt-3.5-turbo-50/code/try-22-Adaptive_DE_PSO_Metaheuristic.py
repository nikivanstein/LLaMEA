import numpy as np

class Adaptive_DE_PSO_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR_min = 0.7
        self.CR_max = 1.0
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1_min = 1.5
        self.c1_max = 2.0
        self.c2_min = 1.5
        self.c2_max = 2.0
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def clip(x):
            return np.clip(x, self.lb, self.ub)

        def evaluate(x):
            return func(clip(x))

        def create_population():
            return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

        def DE(x, target_idx, F, CR):
            a, b, c = np.random.choice(np.delete(np.arange(self.pop_size), target_idx), 3, replace=False)
            mutant = clip(x[a] + F * (x[b] - x[c]))
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, x[target_idx])
            return trial

        def PSO(x, g_best, w, c1, c2):
            v = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
            p_best = x.copy()
            p_best_fitness = np.array([evaluate(xi) for xi in p_best])
            g_best_fitness = min(p_best_fitness)
            g_best = p_best[np.argmin(p_best_fitness)]

            for _ in range(self.max_iter):
                r1, r2 = np.random.rand(2, self.pop_size, self.dim)
                v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
                x = clip(x + v)
                fx = np.array([evaluate(xi) for xi in x])

                for i in range(self.pop_size):
                    if fx[i] < p_best_fitness[i]:
                        p_best[i] = x[i]
                        p_best_fitness[i] = fx[i]
                    if fx[i] < g_best_fitness:
                        g_best = x[i]
                        g_best_fitness = fx[i]

            return g_best

        population = create_population()
        best_solution = population[0]

        for i in range(self.max_iter):
            F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
            CR = self.CR_min + (self.CR_max - self.CR_min) * np.random.rand()
            w = self.w_min + (self.w_max - self.w_min) * np.random.rand()
            c1 = self.c1_min + (self.c1_max - self.c1_min) * np.random.rand()
            c2 = self.c2_min + (self.c2_max - self.c2_min) * np.random.rand()

            for j in range(self.pop_size):
                trial = DE(population, j, F, CR)
                if evaluate(trial) < evaluate(population[j]):
                    population[j] = trial

            best_solution = PSO(population, best_solution, w, c1, c2)

        return best_solution