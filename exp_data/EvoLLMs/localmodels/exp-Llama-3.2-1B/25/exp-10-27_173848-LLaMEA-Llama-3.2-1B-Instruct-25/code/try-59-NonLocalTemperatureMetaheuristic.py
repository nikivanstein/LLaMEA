import numpy as np
import random
from scipy.optimize import minimize

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

    def optimize(self, func, initial_individual, max_iter=1000, tol=1e-6):
        """
        Optimize the black box function using the Non-Local Temperature Metaheuristic.

        Parameters:
        func (function): The black box function to optimize.
        initial_individual (list): The initial individual to start with.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.

        Returns:
        The optimized individual.
        """
        # Create a copy of the initial individual
        individual = initial_individual.copy()

        # Initialize the temperature
        self.temp = 1.0

        # Run the optimization algorithm
        for _ in range(max_iter):
            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual)

            # If the fitness is better, return the individual
            if fitness > 0:
                return individual

            # If the fitness is not better, decrease the temperature
            self.temp *= 0.9

            # Generate a new individual
            new_individual = individual.copy()
            for i in range(self.dim):
                new_individual[i] += np.random.uniform(-1, 1)

            # Evaluate the fitness of the new individual
            new_fitness = self.evaluate_fitness(new_individual)

            # If the fitness is better, update the individual
            if new_fitness > fitness:
                individual = new_individual

        # If the maximum number of iterations is reached, return the best individual found so far
        return individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 