import numpy as np

class DE_PSO_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size
        self.F = 0.5
        self.CR = 0.9
        self.w = 0.5
        self.c1 = 2.0
        self.c2 = 2.0
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def clip(x):
            return np.clip(x, self.lb, self.ub)

        def evaluate(x):
            return func(clip(x))

        def create_population():
            return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

        def DE(x, target_idx):
            a, b, c = np.random.choice(np.delete(np.arange(self.pop_size), target_idx), 3, replace=False)
            mutant = clip(x[a] + self.F * (x[b] - x[c]))
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, x[target_idx])
            return trial

        def PSO(x, g_best):
            v = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
            p_best = x.copy()
            p_best_fitness = np.array([evaluate(xi) for xi in p_best])
            g_best_fitness = min(p_best_fitness)
            g_best = p_best[np.argmin(p_best_fitness)]

            for _ in range(self.max_iter):
                r1, r2 = np.random.rand(2, self.pop_size, self.dim)
                v = self.w * v + self.c1 * r1 * (p_best - x) + self.c2 * r2 * (g_best - x)
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
            for j in range(self.pop_size):
                trial = DE(population, j)
                if evaluate(trial) < evaluate(population[j]):
                    population[j] = trial

            best_solution = PSO(population, best_solution)

        return best_solution