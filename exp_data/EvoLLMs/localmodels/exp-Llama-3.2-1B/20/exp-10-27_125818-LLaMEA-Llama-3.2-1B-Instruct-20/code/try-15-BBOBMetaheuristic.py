# Description: BBOB Metaheuristic with Adaptive Population Size
# Code: 
# ```python
import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim, ps=0.2):
        """
        Initialize the BBOBMetaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        - ps: The proportion of individuals to use for the initial population. Defaults to 0.2.
        """
        self.budget = budget
        self.dim = dim
        self.ps = ps
        self.ps_size = int(self.ps * budget)
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
            # Return the optimized function value
            return self.f

# Description: BBOB Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize
#
# def bboo_metaheuristic(func, budget, dim, ps=0.2):
#     return BBOBMetaheuristic(budget, dim, ps)(func)
#
# def func(x):
#     return x[0]**2 + x[1]**2
#
# budget = 1000
# dim = 2
# ps = 0.2
# metaheuristic = bboo_metaheuristic(func, budget, dim, ps)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')