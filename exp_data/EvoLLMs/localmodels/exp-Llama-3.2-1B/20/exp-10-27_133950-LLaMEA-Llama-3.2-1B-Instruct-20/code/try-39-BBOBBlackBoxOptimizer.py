import numpy as np
import random
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def mutate(self, individual):
        if random.random() < 0.2:
            self.search_space = np.linspace(-5.0, 5.0, 100)
            self.func_evaluations = 0
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.2:
            self.search_space = np.linspace(-5.0, 5.0, 100)
            self.func_evaluations = 0
        child = parent1 + parent2
        if random.random() < 0.2:
            child = parent2 + parent1
        return child

    def __str__(self):
        return f"BBOBBlackBoxOptimizer(budget={self.budget}, dim={self.dim}, population_size={self.population_size})"

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Initialize the population with a random solution
optimizer = BBOBBlackBoxOptimizer(1000, 10)
for _ in range(10):
    optimizer = optimizer.__call__(func)
    print(optimizer)

# Select a random individual from the population
individual = random.choice([optimizer for _ in range(10)])
print(individual)

# Refine the individual using mutation and crossover
individual = optimizer.mutate(individual)
print(individual)

# Refine the individual using crossover
individual = optimizer.crossover(individual, optimizer.func_evaluations % 10)
print(individual)