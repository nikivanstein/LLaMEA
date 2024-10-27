import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = 0.2
        self.beta0 = 1.0
        self.gamma = 0.1

    def __call__(self, func):
        def initialize_population(pop_size, dim, lb, ub):
            return np.random.uniform(lb, ub, (pop_size, dim))

        def attractiveness(distance):
            return self.beta0 * np.exp(-self.gamma * distance**2)

        population = initialize_population(self.population_size, self.dim, self.lb, self.ub)
        for _ in range(self.budget // self.population_size):
            for i, firefly_i in enumerate(population):
                for j, firefly_j in enumerate(population):
                    if func(firefly_j) < func(firefly_i):
                        distance = np.linalg.norm(firefly_i - firefly_j)
                        attractiveness_ij = attractiveness(distance)
                        step_size = attractiveness_ij * (firefly_j - firefly_i)
                        population[i] += self.alpha * step_size
                population[i] = np.clip(population[i], self.lb, self.ub)
        return population[np.argmin([func(p) for p in population])]