import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_black_box(self, func, bounds, initial_values, budget, alpha=0.8):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Parameters:
        func (function): The black box function to optimize.
        bounds (tuple): The search space bounds.
        initial_values (list): The initial values for the search.
        budget (int): The number of function evaluations.
        alpha (float, optional): The mutation probability. Defaults to 0.8.

        Returns:
        list: The optimized values.
        """
        # Initialize the population with random values
        population = initial_values.copy()

        # Evolve the population using differential evolution
        for _ in range(budget):
            # Calculate the fitness of each individual
            fitness = np.array([func(x) for x in population])

            # Select the fittest individuals
            fittest = np.argsort(fitness)[::-1][:int(alpha * budget)]

            # Create a new population with the fittest individuals
            population = [initial_values[i] for i in fittest]

        # Refine the search space
        for _ in range(10):
            # Calculate the fitness of each individual
            fitness = np.array([func(x) for x in population])

            # Select the fittest individuals
            fittest = np.argsort(fitness)[::-1][:int(alpha * budget)]

            # Create a new population with the fittest individuals
            population = [initial_values[i] for i in fittest]

        # Optimize the function values
        optimized_values = np.array([func(x) for x in population])

        # Refine the search space
        for _ in range(10):
            # Calculate the fitness of each individual
            fitness = np.array([func(x) for x in population])

            # Select the fittest individuals
            fittest = np.argsort(fitness)[::-1][:int(alpha * budget)]

            # Create a new population with the fittest individuals
            population = [initial_values[i] for i in fittest]

        # Optimize the function values
        optimized_values = np.array([func(x) for x in population])

        # Refine the search space
        for _ in range(10):
            # Calculate the fitness of each individual
            fitness = np.array([func(x) for x in population])

            # Select the fittest individuals
            fittest = np.argsort(fitness)[::-1][:int(alpha * budget)]

            # Create a new population with the fittest individuals
            population = [initial_values[i] for i in fittest]

        # Optimize the function values
        optimized_values = np.array([func(x) for x in population])

        # Refine the search space
        for _ in range(10):
            # Calculate the fitness of each individual
            fitness = np.array([func(x) for x in population])

            # Select the fittest individuals
            fittest = np.argsort(fitness)[::-1][:int(alpha * budget)]

            # Create a new population with the fittest individuals
            population = [initial_values[i] for i in fittest]

        return population, optimized_values

# Description: Adaptive Black Box Optimization using Differential Evolution.
# Code: 