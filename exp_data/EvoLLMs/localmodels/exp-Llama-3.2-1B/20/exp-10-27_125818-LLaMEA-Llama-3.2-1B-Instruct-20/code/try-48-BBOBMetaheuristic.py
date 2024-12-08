import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBMetaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.space = None
        self.x = None
        self.f = None
        self.population = None
        self.logger = None
        self.refine = False

    def __call__(self, func):
        """
        Optimize the black box function `func` using `self.budget` function evaluations.

        Args:
        - func: The black box function to be optimized.

        Returns:
        - The optimized function value.
        """
        if self.func is None:
            self.func = func
            self.space = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.f = self.func(self.x)
        else:
            while self.budget > 0:
                # Sample a new point in the search space
                self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
                # Evaluate the function at the new point
                self.f = self.func(self.x)
                # Check if the new point is better than the current point
                if self.f < self.f + 1e-6:  # add a small value to avoid division by zero
                    # Update the current point
                    self.x = self.x
                    self.f = self.f
            # If the population is empty, refine the strategy
            if not self.population:
                self.refine = True
                # Refine the strategy by changing the individual lines
                # Update the new individual using adaptive sampling
                self.population = self.refine_sample(self.population, func, self.budget, self.dim)
            # Return the optimized function value
            return self.f

    def refine_sample(self, population, func, budget, dim):
        """
        Refine the strategy by sampling from the population using adaptive sampling.

        Args:
        - population: The current population of individuals.
        - func: The black box function to be optimized.
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.

        Returns:
        - The refined population of individuals.
        """
        # Calculate the average fitness of the population
        avg_fitness = np.mean([self.f(individual, self.logger) for individual in population])
        # Calculate the standard deviation of the fitness
        std_dev = np.std([self.f(individual, self.logger) for individual in population])
        # Refine the strategy by sampling from the population
        refined_population = population[np.random.choice(len(population), budget, replace=False)]
        # Update the refined population
        refined_population = [individual for individual in refined_population if self.f(individual, self.logger) < avg_fitness + 1e-6]
        refined_population = [individual for individual in refined_population if self.f(individual, self.logger) > avg_fitness - 1e-6]
        # Refine the strategy by changing the individual lines
        for individual in refined_population:
            # Update the individual using adaptive sampling
            self.x = self.x + np.random.uniform(-1.0, 1.0, (dim,))
            # Evaluate the function at the new point
            self.f = self.func(self.x)
        # Return the refined population
        return refined_population

# Description: BBOB Metaheuristic with Adaptive Sampling and Refining Strategy
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize
#
# def bboo_metaheuristic(func, budget, dim):
#     return BBOBMetaheuristic(budget, dim)(func)
#
# def func(x):
#     return x[0]**2 + x[1]**2
#
# budget = 1000
# dim = 2
# metaheuristic = bboo_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')