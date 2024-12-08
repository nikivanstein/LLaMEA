import numpy as np

class AdaptiveBBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the AdaptiveBBOBMetaheuristic with a given budget and dimensionality.

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
        self.iterations = 0

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
                if self.iterations % 100 == 0 and self.iterations!= 0:
                    # Refine the strategy by changing the individual lines
                    self.func = self.func.__call__(self.x)
                    self.space = np.random.uniform(-5.0, 5.0, (self.dim,))
                    self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
                    self.f = self.func(self.x)
                # Update the current point
                self.x = self.x
                self.f = self.f
                self.iterations += 1
            # Return the optimized function value
            return self.f

# Description: Adaptive Black Box Optimization using Adaptive Metaheuristic with Refinement
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize
#
# def adaptive_bboo_metaheuristic(func, budget, dim):
#     return AdaptiveBBOBMetaheuristic(budget, dim)(func)
#
# def func(x):
#     return x[0]**2 + x[1]**2
#
# budget = 1000
# dim = 2
# metaheuristic = adaptive_bboo_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')