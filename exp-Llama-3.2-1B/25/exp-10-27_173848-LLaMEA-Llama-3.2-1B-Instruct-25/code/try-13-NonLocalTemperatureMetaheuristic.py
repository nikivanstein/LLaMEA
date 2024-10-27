import numpy as np
import random
import copy
from scipy.optimize import differential_evolution

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            num_evals += 1

        return self.best_func

    def optimize(self, func, bounds, initial_population, budget):
        # Create an initial population
        population = initial_population

        # Run the evolutionary algorithm
        for _ in range(budget):
            # Evaluate the fitness of each individual
            fitness = [self.__call__(func) for individual in population]

            # Select the fittest individuals
            fittest_individuals = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)[:self.budget // 2]

            # Create a new population
            new_population = [copy.deepcopy(individual) for individual, _ in fittest_individuals]

            # Update the population
            population = new_population

        # Return the best individual
        return max(population, key=lambda x: x[0])

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```