import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

def iterated_permutation_cooling(func, budget, dim, cooling_rate=0.95, max_iter=100):
    """
    Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm.

    The algorithm iteratively generates new individuals by iterated permutation and cooling.
    The probability of generating a new individual is proportional to the inverse of the number of evaluations.
    The cooling rate determines the rate at which the algorithm converges to the optimal solution.
    The maximum number of iterations determines the number of times the algorithm generates new individuals.
    """
    population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]  # Initialize the population with 100 individuals
    while len(population) < budget and max_iter > 0:
        new_population = []
        for _ in range(population.size // 2):  # Generate new individuals in half the population
            parent1, parent2 = random.sample(population, 2)
            child = (parent1 + parent2) / 2
            new_population.append(child)
        population = new_population
        for individual in population:
            func_value = func(individual)
            if func_value > np.max([func(np.random.uniform(-5.0, 5.0, dim)) for np.random.uniform(-5.0, 5.0, dim) in population])):
                func_value = func(individual)
        if np.random.rand() < cooling_rate:
            # If the algorithm has not converged, generate a new individual by iterated permutation
            permuted_population = [np.random.permutation(individual) for individual in population]
            new_population = []
            for _ in range(population.size // 2):
                parent1, parent2 = random.sample(permuted_population, 2)
                child = (parent1 + parent2) / 2
                new_population.append(child)
            population = new_population
        max_iter -= 1
    return np.max([func(individual) for individual in population])

# Example usage:
budget = 100
dim = 10
func = lambda x: x**2
optimizer = BlackBoxOptimizer(budget, dim)
best_func_value = -np.inf
best_func = None
for _ in range(1000):
    func_value = iterated_permutation_cooling(func, budget, dim)
    if func_value > best_func_value:
        best_func_value = func_value
        best_func = func_value
print("Best function value:", best_func_value)
print("Best function:", best_func)