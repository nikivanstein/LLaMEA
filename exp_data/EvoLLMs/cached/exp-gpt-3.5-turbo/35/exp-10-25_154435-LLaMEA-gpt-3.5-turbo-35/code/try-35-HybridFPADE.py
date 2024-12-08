import numpy as np

class HybridFPADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iters = budget // self.population_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population(dim, population_size, lb, ub):
            return np.random.uniform(lb, ub, (population_size, dim))

        def clip_position(p, lb, ub):
            return np.clip(p, lb, ub)

        def optimize(func, population):
            best_pos = None
            best_val = np.inf
            for i in range(len(population)):
                fitness = func(population[i])
                if fitness < best_val:
                    best_val = fitness
                    best_pos = population[i]
            return best_pos

        population = initialize_population(self.dim, self.population_size, self.lb, self.ub)
        for _ in range(self.max_iters):
            for i in range(self.population_size):
                p_best = optimize(func, population)
                r1, r2 = np.random.uniform(0, 1, (2, self.dim))
                v = population[i] + 0.5 * (population[i] - population[i]) + 0.5 * (p_best - population[i]) + 0.5 * (r1 - r2)
                population[i] = clip_position(v, self.lb, self.ub)
        return optimize(func, population)