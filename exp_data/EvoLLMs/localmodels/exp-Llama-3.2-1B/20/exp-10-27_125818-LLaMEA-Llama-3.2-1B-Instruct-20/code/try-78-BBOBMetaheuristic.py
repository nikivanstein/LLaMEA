# Description: BBOBMetaheuristic with Adaptive Refining
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
        self.refining = False
        self.refining_interval = 1000

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
                # Refine the strategy if necessary
                if self.refining and self.budget > 0:
                    # Sample a new point in the search space
                    self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
                    # Evaluate the function at the new point
                    self.f = self.func(self.x)
                    # Check if the new point is better than the current point
                    if self.f < self.f + 1e-6:  # add a small value to avoid division by zero
                        # Update the current point
                        self.x = self.x
                        self.f = self.f
                        self.refining = False
                        # Refine the strategy every 1000 evaluations
                        if self.budget > self.refining_interval:
                            self.refining = True
            # Return the optimized function value
            return self.f

# Description: BBOBMetaheuristic with Adaptive Refining
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