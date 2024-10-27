import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.func_evals = 0

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(*self.search_space, size=self.search_space_dim) for _ in range(100)]

        # Initialize the best individual and its fitness
        best_individual = population[0]
        best_fitness = self.func_evals_evals(population[0], func)

        # Evaluate the function for all individuals
        for individual in population:
            func_value = self.func_evals_evals(individual, func)
            # Update the best individual if the current individual is better
            if func_value > best_fitness:
                best_individual = individual
                best_fitness = func_value

        # Evaluate the function at the final point in the search space
        func_value = self.func_evals_evals(best_individual, func)
        return best_individual, func_value

    def func_evals_evals(self, individual, func):
        # Evaluate the function at the individual using the budget function evaluations
        evaluations = np.random.uniform(*self.search_space, size=self.search_space_dim)
        return func(evaluations)

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1, optimized_func2 = hyperband(test_func1)
optimized_func3, optimized_func4 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2, optimized_func3, optimized_func4], label=['Test Function 1', 'Test Function 2', 'Test Function 3', 'Test Function 4'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()

# Refine the strategy
def refine_strategy(individual, func):
    # Initialize the population with random points in the search space
    population = [np.random.uniform(*self.search_space, size=self.search_space_dim) for _ in range(100)]

    # Initialize the best individual and its fitness
    best_individual = individual
    best_fitness = func(individual)

    # Evaluate the function for all individuals
    for individual in population:
        func_value = func(individual)
        # Update the best individual if the current individual is better
        if func_value > best_fitness:
            best_individual = individual
            best_fitness = func_value

    # Evaluate the function at the final point in the search space
    func_value = func(best_individual)
    return best_individual, func_value

optimized_func1, optimized_func2 = refine_strategy(optimized_func1, test_func1)
optimized_func3, optimized_func4 = refine_strategy(optimized_func3, test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2, optimized_func3, optimized_func4], label=['Refined Test Function 1', 'Refined Test Function 2', 'Refined Test Function 3', 'Refined Test Function 4'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Refined Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()