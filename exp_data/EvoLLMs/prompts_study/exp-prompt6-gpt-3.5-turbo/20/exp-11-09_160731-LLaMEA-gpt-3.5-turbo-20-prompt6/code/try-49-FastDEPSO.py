import numpy as np

class FastDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w = 0.5
        self.c1 = 1.494
        self.c2 = 1.494
        self.cr = 0.9
        self.mutation_prob = 0.5
        self.mutation_step = 1.0

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

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.uniform(size=2)
                v = population[i] + self.w * (p_best - population[i]) + self.c1 * r1 * (g_best - population[i])
                u = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.uniform() < self.cr or j == j_rand:
                        u[j] = v[j]
                        if np.random.uniform() < self.mutation_prob:
                            u[j] += np.random.uniform(-self.mutation_step, self.mutation_step)
                u = clipToBounds(u)
                if func(u) < func(population[i]):
                    population[i] = u
                    if func(u) < func(p_best):
                        p_best = u
                if func(u) < func(g_best):
                    g_best = u
                    self.mutation_prob *= 1.05 if func(u) < func(g_best) else 0.95
                    self.mutation_step *= 1.03 if func(u) < func(g_best) else 0.97  # Dynamic adjustment of mutation step size

        return g_best