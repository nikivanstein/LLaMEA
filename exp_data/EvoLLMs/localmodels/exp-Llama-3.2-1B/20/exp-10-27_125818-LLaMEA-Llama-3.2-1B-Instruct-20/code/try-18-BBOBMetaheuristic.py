# Description: Adaptive BBOB Metaheuristic
# Code: 
# ```python
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
        self.population_history = []

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
            self.population = [self.x]
            self.logger = None
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
                # Store the new point in the population history
                self.population_history.append(self.x)
                # Check if the population has reached its maximum size
                if len(self.population) > self.budget:
                    # Refine the strategy by changing the individual lines of the selected solution
                    self.x[0] = np.random.uniform(-5.0, 5.0)
                    self.x[1] = np.random.uniform(-5.0, 5.0)
                    self.f = self.func(self.x)
                # Update the population with the new individual
                self.population.append(self.x)
                # Check if the population has reached its maximum size
                if len(self.population) > self.budget:
                    # Refine the strategy by changing the individual lines of the selected solution
                    self.x = self.population_history[-1]
                    self.f = self.func(self.x)
                # Check if the population has reached its maximum size
                if len(self.population) > self.budget:
                    # Refine the strategy by changing the individual lines of the selected solution
                    self.x = self.population_history[-1]
                    self.f = self.func(self.x)
                # Return the optimized function value
                return self.f

# Description: Adaptive BBOB Metaheuristic
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

# Refining the strategy by changing the individual lines of the selected solution
class BBOBMetaheuristicWithRefiningStrategy(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.refining_strategy = False

    def __call__(self, func):
        """
        Optimize the black box function `func` using `self.budget` function evaluations.

        Args:
        - func: The black box function to be optimized.

        Returns:
        - The optimized function value.
        """
        if self.refining_strategy:
            # Refine the strategy by changing the individual lines of the selected solution
            self.x[0] = np.random.uniform(-5.0, 5.0)
            self.x[1] = np.random.uniform(-5.0, 5.0)
            self.f = self.func(self.x)
            self.refining_strategy = False
        else:
            super().__call__(func)
            self.refining_strategy = True

# Description: Adaptive BBOB Metaheuristic with refining strategy
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

# Test the refined strategy
metaheuristic = BBOBMetaheuristicWithRefiningStrategy(budget=1000, dim=2)
x0 = [1.0, 1.0]
res = metaheuristic(func, budget=1000, dim=2)
print(f'Optimized function: {res.fun}')
print(f'Optimized parameters: {res.x}')