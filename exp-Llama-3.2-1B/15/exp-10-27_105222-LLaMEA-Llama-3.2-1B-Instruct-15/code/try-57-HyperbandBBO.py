import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

    def bayesian_optimization(self, func, initial_point, num_iterations, alpha, beta):
        # Initialize the population with random points
        population = [initial_point]
        
        # Run the optimization algorithm for the specified number of iterations
        for _ in range(num_iterations):
            # Evaluate the fitness of each individual in the population
            fitness_values = [func(individual) for individual in population]
            
            # Select the fittest individuals to reproduce
            fittest_indices = np.argsort(fitness_values)[-self.budget:]
            fittest_individuals = [population[i] for i in fittest_indices]
            
            # Create new offspring by perturbing the fittest individuals
            offspring = []
            for _ in range(self.budget):
                individual = np.random.choice(fittest_individuals, size=self.dim, replace=False)
                perturbation = np.random.normal(0, 1, size=self.dim)
                offspring.append(individual + perturbation)
            
            # Evaluate the fitness of the new offspring
            new_fitness_values = [func(individual) for individual in offspring]
            
            # Select the fittest new offspring to reproduce
            new_fittest_indices = np.argsort(new_fitness_values)[-self.budget:]
            new_fittest_individuals = [offspring[i] for i in new_fittest_indices]
            
            # Replace the population with the new offspring
            population = new_fittest_individuals
        
        # Return the fittest individual in the final population
        return population[0]

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

# One-line description with the main idea
# Description: Efficient Black Box Optimization using Hyperband and Bayesian Optimization
# Code: 
# ```python
# HyperbandBBO
# ```
# ```python
# def bayesian_optimization(self, func, initial_point, num_iterations, alpha, beta):
#     population = [initial_point]
#     for _ in range(num_iterations):
#         fitness_values = [func(individual) for individual in population]
#         fittest_indices = np.argsort(fitness_values)[-self.budget:]
#         fittest_individuals = [population[i] for i in fittest_indices]
#         offspring = []
#         for _ in range(self.budget):
#             individual = np.random.choice(fittest_individuals, size=self.dim, replace=False)
#             perturbation = np.random.normal(0, 1, size=self.dim)
#             offspring.append(individual + perturbation)
#         new_fitness_values = [func(individual) for individual in offspring]
#         new_fittest_indices = np.argsort(new_fitness_values)[-self.budget:]
#         new_fittest_individuals = [offspring[i] for i in new_fittest_indices]
#         population = new_fittest_individuals
#     return population[0]