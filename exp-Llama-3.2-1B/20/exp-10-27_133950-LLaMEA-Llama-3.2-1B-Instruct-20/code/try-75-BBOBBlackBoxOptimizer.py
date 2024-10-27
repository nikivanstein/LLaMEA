import numpy as np
from scipy.optimize import minimize
from collections import deque

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population = deque(maxlen=self.budget)

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
        if len(individual) < self.dim:
            return individual + [np.random.uniform(-5.0, 5.0) for _ in range(self.dim - len(individual))]
        else:
            return individual

    def __str__(self):
        return f"BBOBBlackBoxOptimizer(budget={self.budget}, dim={self.dim})"

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Mutate the solution
mutated_individual = optimizer.mutate(result)
print(mutated_individual)

# Evaluate the mutated solution
new_individual = optimizer(func)(mutated_individual)
print(new_individual)

# Update the population
optimizer.population.append(new_individual)
print(optimizer.population)

# Select the best individual
best_individual = min(optimizer.population, key=optimizer.__call__)
print(best_individual)

# Print the updated population
print(optimizer.population)