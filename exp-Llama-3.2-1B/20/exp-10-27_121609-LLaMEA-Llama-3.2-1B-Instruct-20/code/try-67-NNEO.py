# Black Box Optimization using Evolutionary Strategies
# Description: This algorithm uses evolutionary strategies to optimize black box functions with a specified budget.

import numpy as np
from scipy.optimize import minimize

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

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, individual):
        new_individual = individual.copy()
        if np.random.rand() < 0.2:  # Change strategy
            new_individual = np.clip(new_individual, -5.0, 5.0)
        return new_individual

    def evaluate_fitness(self, individual):
        fitness = minimize(objective, individual, bounds=bounds, method="SLSQP", options={"xatol": 1e-6})
        return fitness.fun

# Example usage:
nneo = NNEO(100, 10)
func = lambda x: x**2
best_individual = nneo(func)
best_fitness = nneo.evaluate_fitness(best_individual)
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")