import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func, bounds, mutation_rate, mutation_probability):
        def objective(x):
            return func(x)

        def bounds_check(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if np.random.rand() < mutation_probability:
                return x + np.random.uniform(-5.0, 5.0)
            else:
                return x

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = mutate(x)

        return self.fitnesses

# Novel Metaheuristic Algorithm
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 