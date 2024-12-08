import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_lower = self.search_space[0]
        self.search_space_upper = self.search_space[1]

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = norm.rvs(loc=self.search_space_lower, scale=self.search_space_upper - self.search_space_lower, size=self.search_space_dim)
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

    def bayesian_optimization(self, func, num_samples):
        # Initialize the search space
        search_space = self.search_space
        # Initialize the population
        population = [self.bayesian_optimization_func(func, search_space, num_samples) for _ in range(100)]
        # Initialize the best individual
        best_individual = None
        best_fitness = float('-inf')
        # Perform Bayesian optimization
        for i in range(100):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual) for individual in population]
            # Get the index of the individual with the best fitness
            idx = np.argmin(fitness)
            # Update the best individual and its fitness
            best_individual = population[idx]
            best_fitness = fitness[idx]
            # Update the search space
            search_space = (best_individual[0] - 0.1, best_individual[0] + 0.1)
        # Return the best individual and its fitness
        return best_individual, best_fitness

    def bayesian_optimization_func(self, func, search_space, num_samples):
        # Initialize the population
        population = [func(x) for x in np.random.uniform(search_space[0], search_space[1], size=(num_samples, self.dim))]
        # Initialize the best individual
        best_individual = None
        best_fitness = float('-inf')
        # Perform Bayesian optimization
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual) for individual in population]
            # Get the index of the individual with the best fitness
            idx = np.argmin(fitness)
            # Update the best individual and its fitness
            best_individual = population[idx]
            best_fitness = fitness[idx]
            # Update the search space
            search_space = (best_individual[0] - 0.1, best_individual[0] + 0.1)
        # Return the best individual and its fitness
        return best_individual, best_fitness

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()

# Bayesian optimization
best_individual, best_fitness = hyperband.bayesian_optimization(test_func1, 100)
print(f'Best individual: {best_individual}, Best fitness: {best_fitness}')