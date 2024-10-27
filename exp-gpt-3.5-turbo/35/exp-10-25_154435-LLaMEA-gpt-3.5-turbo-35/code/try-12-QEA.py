import numpy as np
from scipy.stats import norm

class QEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iters = budget // self.population_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population(dim, pop_size, lb, ub):
            return np.random.uniform(lb, ub, (pop_size, dim))

        def clip_position(p, lb, ub):
            return np.clip(p, lb, ub)

        def fitness_score(population, func):
            return np.array([func(p) for p in population])

        population = initialize_population(self.dim, self.population_size, self.lb, self.ub)
        for _ in range(self.max_iters):
            fitness = fitness_score(population, func)
            fitness_prob = norm.cdf(fitness)
            parents = population[np.random.rand(self.population_size) < fitness_prob]
            children = np.repeat(parents, 2, axis=0)
            noise = np.random.normal(0, 0.1, children.shape)
            children += noise
            population[:len(children)] = clip_position(children, self.lb, self.ub)
        best_solution = population[np.argmin(fitness_score(population, func))]
        return best_solution