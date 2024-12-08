import numpy as np
from scipy.optimize import minimize

class EvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mu = 0.5
        self.sigma = 1.0
        self.population_size = 100

    def __call__(self, func):
        def objective(x):
            return func(x)

        def fitness(x):
            return -objective(x)

        def mutate(x):
            return x + np.random.normal(0, self.sigma, self.dim)

        def evolve(x):
            new_x = np.array([mutate(x[i]) for i in range(self.population_size)])
            new_x = np.array([minimize(fitness, x_i).x for x_i in new_x])
            return new_x

        # Initialize population
        x = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            x = evolve(x)

        # Select best solution
        best_idx = np.argmin([objective(x[i]) for i in range(self.population_size)])
        best_x = x[best_idx]

        # Refine strategy
        refine_prob = 0.3
        for i in range(self.population_size):
            if np.random.rand() < refine_prob:
                best_x[i] = mutate(best_x[i])

        return best_x

# Example usage
budget = 100
dim = 10
es = EvolutionStrategy(budget, dim)
best_x = es(lambda x: x**2)
print(best_x)