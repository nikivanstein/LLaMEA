import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func, algorithm='bbo'):
        if algorithm == 'bbo':
            return self.bbo(func)
        elif algorithm == 'hyperband':
            return self.hyperband(func)
        else:
            raise ValueError("Invalid algorithm. Choose 'bbo' or 'hyperband'.")

    def bbo(self, func):
        # Initialize population with random points in the search space
        population = [random.uniform(*self.search_space) for _ in range(100)]

        # Define the budget for each individual
        budget = self.budget / len(population)

        # Run the algorithm for the specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)]

            # Select a subset of individuals for the next generation
            next_generation = fittest_individuals[:int(len(fittest_individuals) * 0.7)]

            # Update the population with the next generation
            population = next_generation

            # Update the search space
            self.search_space = (min(self.search_space[0], population.min()), max(self.search_space[1], population.max()))

        # Evaluate the fitness of the final population
        fitness = [self.evaluate_fitness(individual, func) for individual in population]
        return population[np.argsort(fitness)]

    def hyperband(self, func):
        # Initialize the population with random points in the search space
        population = [random.uniform(*self.search_space) for _ in range(100)]

        # Define the budget for each individual
        budget = self.budget / len(population)

        # Run the algorithm for the specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)]

            # Select a subset of individuals for the next generation
            next_generation = fittest_individuals[:int(len(fittest_individuals) * 0.7)]

            # Update the population with the next generation
            population = next_generation

            # Update the search space
            self.search_space = (min(self.search_space[0], population.min()), max(self.search_space[1], population.max()))

        # Evaluate the fitness of the final population
        fitness = [self.evaluate_fitness(individual, func) for individual in population]
        return population[np.argsort(fitness)]

    def evaluate_fitness(self, individual, func):
        return func(individual)

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