import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import copy

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
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

    def mutate(self, individual):
        # Randomly change one gene in the individual
        gene_index = random.randint(0, self.dim - 1)
        self.search_space[gene_index] = random.uniform(self.search_space[gene_index], 5.0)
        return individual

    def bayes_optimize(self, func, bounds, initial_points, alpha):
        # Initialize the population with random points in the search space
        population = [copy.deepcopy(initial_points)]
        for _ in range(self.budget):
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]
            # Select the fittest individuals
            selected_indices = np.argsort(fitnesses)[:self.dim]
            # Create a new population with the selected individuals
            new_population = [individual for index, individual in enumerate(population) if index in selected_indices]
            # Add new individuals to the population
            new_population += [individual for individual in population if index not in selected_indices]
            # Update the population with the new individuals
            population = new_population
            # Update the search space
            new_search_space = (min(bounds[0][0], self.search_space[0]), max(bounds[0][1], self.search_space[1]))
            # Update the bounds
            bounds = (new_search_space, new_search_space)
        # Return the best individual in the population
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