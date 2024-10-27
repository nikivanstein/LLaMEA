import numpy as np

class WhaleOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.lb = -5.0
        self.ub = 5.0
        self.a = 2  # Parameter for spiral updating
        self.a_max = 2  # Maximum value for parameter a
        self.b = 1  # Parameter for spiral updating
        self.max_iters = 100

    def __call__(self, func):
        def initialize_population(population_size, dim, lb, ub):
            return np.random.uniform(lb, ub, (population_size, dim))

        def levy_flight():
            beta = 1.5
            sigma1 = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            sigma2 = 1
            u = np.random.normal(0, sigma1, 1)[0]
            v = np.random.normal(0, sigma2, 1)[0]
            step = u / abs(v) ** (1 / beta)
            return step

        population = initialize_population(self.population_size, self.dim, self.lb, self.ub)
        g_best = population[np.argmin([func(p) for p in population])]
        for _ in range(self.budget // self.population_size):
            a = self.a - (self.a / self.max_iters) * _  # Update parameter a
            for i in range(self.population_size):
                r = np.random.rand()  # Random number
                A = 2 * a * r - a  # Parameter A
                distance_to_g_best = np.abs(g_best - population[i])
                if r < 0.5:
                    if np.linalg.norm(distance_to_g_best) < 1:
                        population[i] = g_best - A * distance_to_g_best
                    else:
                        population[i] = g_best - A * levy_flight()
                else:
                    population[i] = np.random.uniform(self.lb, self.ub, self.dim)
            g_best = population[np.argmin([func(p) for p in population])]
        return g_best