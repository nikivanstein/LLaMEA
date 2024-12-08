import numpy as np

class FastDEPSO(ImprovedDEPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_step = 0.8

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
                    self.mutation_step *= 0.8 if func(u) < func(g_best) else 1.2

        return g_best