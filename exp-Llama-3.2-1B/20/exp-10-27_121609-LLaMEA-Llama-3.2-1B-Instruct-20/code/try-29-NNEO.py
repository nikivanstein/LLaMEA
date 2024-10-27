import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(x):
            if random.random() < 0.2:
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

class BBBOptimizer:
    def __init__(self, algorithm, budget, dim):
        self.algorithm = algorithm
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        return self.algorithm(func)

# Example usage
def black_box_function(x):
    return np.sin(x)

optimizer = BBBOptimizer(NNEO, 100, 10)
optimized_func = optimizer(black_box_function)

print("Optimized function:", optimized_func)
print("Fitness:", optimized_func[-1])