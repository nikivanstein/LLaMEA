import numpy as np
from scipy.optimize import minimize

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

    def adaptive_black_box(self, func, initial_values, bounds, max_iter=100, tol=1e-6):
        """
        Adaptive Black Box Optimization Algorithm.

        Parameters:
        func (function): The black box function to optimize.
        initial_values (array): The initial values for the optimization.
        bounds (array): The bounds for each dimension.
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

        Returns:
        array: The optimized values.
        """
        # Initialize the population with random values
        population = initial_values.copy()
        for _ in range(10):  # Initial population size
            population = np.random.uniform(bounds[0], bounds[1], size=population.shape)
        
        # Run the evolutionary algorithm
        for _ in range(max_iter):
            # Evaluate the fitness of each individual
            fitness = np.array([func(x) for x in population])
            # Select the fittest individuals
            fittest = np.argsort(fitness)[::-1][:self.budget]
            # Create a new population by crossover and mutation
            population = []
            for _ in range(self.budget):
                parent1, parent2 = fittest.pop(0), fittest.pop(0)
                child = (parent1 + parent2) / 2
                if np.random.rand() < 0.5:
                    child += np.random.uniform(bounds[0], bounds[1])
                population.append(child)
        
        # Evaluate the fitness of the final population
        fitness = np.array([func(x) for x in population])
        # Select the fittest individuals
        fittest = np.argsort(fitness)[::-1][:self.budget]
        # Return the optimized values
        return population[fittest]