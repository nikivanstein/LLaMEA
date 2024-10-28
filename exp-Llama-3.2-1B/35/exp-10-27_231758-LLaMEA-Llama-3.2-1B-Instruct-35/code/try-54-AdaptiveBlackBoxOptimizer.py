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

    def adaptive_black_box(self, func, bounds, max_iter=100, tol=1e-6):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Args:
            func (callable): The function to optimize.
            bounds (list): A list of tuples representing the bounds for each dimension.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

        Returns:
            tuple: A tuple containing the optimized function values and the optimization time.
        """
        # Initialize the population with random values
        population = [random.uniform(bounds[0][0], bounds[0][1]) for _ in range(self.dim)]

        # Run the optimization algorithm
        for _ in range(max_iter):
            # Evaluate the function at each individual in the population
            func_values = np.array([func(value) for value in population])

            # Select the fittest individuals
            fittest_individuals = population[np.argsort(func_values)]

            # Create a new population by crossover and mutation
            new_population = []
            while len(new_population) < self.dim:
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = (parent1 + parent2) / 2
                if random.random() < 0.5:
                    # Mutation: swap two random elements
                    idx1 = np.random.randint(0, self.dim)
                    idx2 = np.random.randint(0, self.dim)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

            # Check for convergence
            if np.allclose(func_values, population):
                break

        # Return the optimized function values and the optimization time
        return func_values, max_iter

    def run(self, func, bounds, max_iter=100, tol=1e-6):
        """
        Run the Adaptive Black Box Optimization algorithm.

        Args:
            func (callable): The function to optimize.
            bounds (list): A list of tuples representing the bounds for each dimension.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

        Returns:
            tuple: A tuple containing the optimized function values and the optimization time.
        """
        # Initialize the population with random values
        population = [random.uniform(bounds[0][0], bounds[0][1]) for _ in range(self.dim)]

        # Run the optimization algorithm
        func_values, time = self.adaptive_black_box(func, bounds, max_iter, tol)

        # Return the optimized function values and the optimization time
        return func_values, time