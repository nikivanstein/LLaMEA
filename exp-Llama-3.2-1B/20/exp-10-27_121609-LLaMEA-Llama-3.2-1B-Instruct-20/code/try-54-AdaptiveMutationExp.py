import numpy as np

class AdaptiveMutationExp:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if np.random.rand() < self.mutation_rate:
                idx = np.random.randint(0, self.dim)
                x[idx] = np.random.uniform(bounds[x][0], bounds[x][1])
            return x

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = mutate(x)

        return self.fitnesses

class NNEO(AdaptiveMutationExp):
    def __init__(self, budget, dim):
        super().__init__(budget, dim, 0.2)

    def __call__(self, func):
        return super().__call__(func)

# One-line description with main idea
# Evolutionary algorithm with adaptive mutation rate to optimize black box functions