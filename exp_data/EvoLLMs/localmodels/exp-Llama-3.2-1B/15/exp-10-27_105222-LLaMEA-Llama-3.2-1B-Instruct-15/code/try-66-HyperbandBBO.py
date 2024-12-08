# Code:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import copy

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func, algorithm):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

    def bayesian_optimization(self, func, algorithm, num_samples):
        # Initialize the population
        population = [copy.deepcopy(func) for _ in range(100)]

        # Run the algorithm for a fixed number of iterations
        for _ in range(10):
            # Select the best individual
            best_individual = max(population, key=algorithm)

            # Select a random subset of the remaining individuals
            random_subset = random.sample(population, len(population) - 1)

            # Evaluate the fitness of each individual
            fitness = [self.__call__(func, individual) for individual in random_subset]

            # Select the best individual based on the Bayesian optimization algorithm
            best_individual = self.bayesian_selection(population, fitness, num_samples)

            # Update the population
            population = [best_individual]

        # Evaluate the fitness of the final population
        fitness = [self.__call__(func, individual) for individual in population]
        best_individual = max(population, key=fitness)
        return best_individual

    def bayesian_selection(self, population, fitness, num_samples):
        # Select the best individual based on the Bayesian optimization algorithm
        # This is a simplified version of the Bayesian optimization algorithm
        # In a real-world scenario, you would need to implement a more complex algorithm
        # that takes into account the uncertainty of the function evaluations
        return population[np.argmax(fitness)]

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1, HyperbandBBO.bayesian_optimization)
optimized_func2 = hyperband(test_func2, HyperbandBBO.bayesian_optimization)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()