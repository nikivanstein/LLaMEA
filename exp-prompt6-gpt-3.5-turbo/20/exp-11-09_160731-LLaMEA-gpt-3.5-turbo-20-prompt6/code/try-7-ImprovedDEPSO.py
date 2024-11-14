import numpy as np

class ImprovedDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 1.494
        self.c2 = 1.494
        self.cr = 0.9

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))

        def clipToBounds(population):
            return np.clip(population, self.lower_bound, self.upper_bound)

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        population = initialize_population()
        population_fitness = evaluate_population(population)
        p_best = population[np.argmin(population_fitness)]
        g_best = p_best
        w_min = 0.4
        w_max = 0.9

        for _ in range(self.max_iter):
            w = w_max - (_ / self.max_iter) * (w_max - w_min)
            for i in range(self.pop_size):
                r1, r2 = np.random.uniform(size=2)
                v = population[i] + w * (p_best - population[i]) + self.c1 * r1 * (g_best - population[i])
                u = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.uniform() < self.cr or j == j_rand:
                        u[j] = v[j]
                u = clipToBounds(u)
                if func(u) < func(population[i]):
                    population[i] = u
                    if func(u) < func(p_best):
                        p_best = u
                if func(u) < func(g_best):
                    g_best = u

        return g_best