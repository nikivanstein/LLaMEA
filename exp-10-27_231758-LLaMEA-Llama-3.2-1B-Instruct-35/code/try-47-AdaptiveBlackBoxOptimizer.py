import numpy as np
import random
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

    def adaptive_black_box(self, func, initial_values, budget, step_size=0.1):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Parameters:
        func (function): The objective function to optimize.
        initial_values (array): The initial values of the variables.
        budget (int): The number of function evaluations allowed.
        step_size (float, optional): The step size for the search. Defaults to 0.1.

        Returns:
        array: The optimized values.
        """
        # Initialize the population with random values
        population = initial_values + np.random.uniform(-5.0, 5.0, size=self.dim)

        # Evolve the population using Differential Evolution
        for _ in range(self.budget):
            # Calculate the fitness of each individual
            fitness = [func(x) for x in population]

            # Select the fittest individuals
            fittest = np.argsort(fitness)[-self.budget:]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for _ in range(self.dim):
                idx = random.choice(fittest)
                new_individual = population[idx] + step_size * (random.uniform(-5.0, 5.0) - 5.0)
                new_population.append(new_individual)

            # Replace the old population with the new one
            population = new_population

        # Return the optimized values
        return np.array(population)

# Description: Adaptive Black Box Optimization using Differential Evolution
# Code: 