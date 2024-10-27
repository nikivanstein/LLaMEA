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
        self.search_space_bounds = self._create_search_space_bounds()

    def _create_search_space_bounds(self):
        return ((self.search_space[0] - self.search_space_bounds[0]) / 2, (self.search_space[1] + self.search_space_bounds[1]) / 2)

    def _gaussian_search(self, x, func):
        # Sample a new point in the search space using Gaussian distribution
        mean, cov = self.search_space_bounds
        sigma = np.sqrt(2 * self.budget / len(x))
        x_new = x + norm.rvs(loc=mean, scale=sigma, size=len(x))
        # Evaluate the function at the new point
        func_value = func(x_new)
        return func_value

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = np.random.uniform(*self.search_space_bounds, size=self.search_space_dim * self.dim)
        population = np.stack([population, population]).T

        while self.func_evals < self.budget:
            # Evaluate the fitness of each individual in the population
            fitnesses = self.evaluate_fitness(population)
            # Select the fittest individuals
            fittest_individuals = np.argsort(fitnesses)[::-1][:self.dim]
            # Create a new population with the fittest individuals
            new_population = population[fittest_individuals]
            new_population = new_population[:, fittest_individuals]

            # Sample a new point in the search space using Gaussian distribution
            new_x = self._gaussian_search(new_population, func)

            # Evaluate the function at the new point
            new_func_value = func(new_x)
            # Store the new point and its fitness value
            new_population = np.stack([new_x, new_func_value]).T
            self.func_evals += 1

        # Evaluate the function at the final point in the search space
        new_func_value = func(new_population)
        return new_func_value

    def evaluate_fitness(self, population):
        # Evaluate the fitness of each individual in the population
        fitnesses = np.zeros(len(population))
        for i, individual in enumerate(population):
            fitnesses[i] = self._gaussian_search(individual, self.func)
        return fitnesses

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