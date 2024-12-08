# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import numpy as np
import random

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
            # Refine the strategy using evolutionary algorithms
            self.x, self.f = self.refine_strategy(self.x, self.f)
        # Return the optimized function value
        return self.f

    def refine_strategy(self, x, f):
        """
        Refine the strategy using evolutionary algorithms.

        Args:
        - x: The current point.
        - f: The function value at the current point.

        Returns:
        - The refined point and function value.
        """
        # Define the mutation rate and selection parameters
        mutation_rate = 0.01
        selection_rate = 0.2

        # Initialize the population
        population = [x]

        # Generate the population for the specified number of generations
        for _ in range(100):
            # Select the fittest individuals
            fittest = sorted(population, key=lambda x: x[-1], reverse=True)[:int(selection_rate * len(population))]

            # Create a new population by mutating the fittest individuals
            new_population = []
            for _ in range(len(fittest)):
                # Mutate the individual
                mutated = [x + random.uniform(-mutation_rate, mutation_rate) for x in fittest]
                # Select the fittest individual
                selected = sorted(mutated, key=lambda x: x[-1], reverse=True)[:int(selection_rate * len(mutated))]
                # Add the mutated individual to the new population
                new_population.extend(selected)

            # Replace the old population with the new population
            population = new_population

            # Update the function value for the new individuals
            for i in range(len(new_population)):
                new_population[i] = self.func(new_population[i])

        # Return the refined point and function value
        return x, new_population[-1][-1]

# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize

# def bboo_metaheuristic(func, budget, dim):
#     return BBOBMetaheuristic(budget, dim)(func)

# def func(x):
#     return x[0]**2 + x[1]**2

# budget = 1000
# dim = 2
# metaheuristic = bboo_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')

# Refine the strategy using evolutionary algorithms
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize

# def bboo_metaheuristic(func, budget, dim):
#     return BBOBMetaheuristic(budget, dim)(func)

# def func(x):
#     return x[0]**2 + x[1]**2

# budget = 1000
# dim = 2
# metaheuristic = bboo_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# refined_x, refined_f = metaheuristic.refine_strategy(x0, res.fun)
# print(f'Optimized function: {refined_f}')
# print(f'Optimized parameters: {refined_x}')