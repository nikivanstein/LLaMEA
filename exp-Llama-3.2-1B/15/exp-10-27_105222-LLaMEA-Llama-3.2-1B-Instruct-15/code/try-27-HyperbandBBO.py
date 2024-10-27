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

    def __call__(self, func):
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

    def bayesian_optimization(self, func, num_samples, alpha=0.1):
        # Initialize the population with random points in the search space
        population = np.random.uniform(*self.search_space, size=(num_samples, self.dim))
        
        # Define the fitness function to evaluate the population
        def fitness(individual):
            return func(individual)
        
        # Define the bounds for the individual points
        bounds = tuple((min(self.search_space[0], self.search_space[1])) for _ in range(self.dim))
        
        # Run the Bayesian optimization algorithm
        results = []
        for _ in range(10):  # Run 10 iterations
            # Evaluate the fitness of each individual in the population
            fitness_values = [fitness(individual) for individual in population]
            
            # Get the index of the individual with the highest fitness value
            idx = np.argmax(fitness_values)
            
            # Update the population with the new individual
            population[idx] = np.random.uniform(*bounds, size=self.dim)
        
        # Return the best individual found
        return population[np.argmax(fitness_values)]

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
best_individual = hyperband.bayesian_optimization(test_func1, num_samples=1000)
print("Bayesian optimization result:", best_individual)