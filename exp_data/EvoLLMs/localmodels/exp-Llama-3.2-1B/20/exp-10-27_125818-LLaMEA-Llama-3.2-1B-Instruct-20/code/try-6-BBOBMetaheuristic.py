import numpy as np
from scipy.optimize import minimize

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
            # Return the optimized function value
            return self.f

    def __str__(self):
        return f"BBOB Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems."

    def __str__(self):
        return "BBOB Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems."

# Description: BBOB Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
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

# Refine the strategy by introducing a probability of changing the individual line
def bboo_metaheuristic_refined(func, budget, dim, prob_change):
    """
    Refine the strategy of the BBOB Metaheuristic by introducing a probability of changing the individual line.

    Args:
    - func: The black box function to be optimized.
    - budget: The maximum number of function evaluations allowed.
    - dim: The dimensionality of the optimization problem.
    - prob_change: The probability of changing the individual line.

    Returns:
    - The refined BBOB Metaheuristic.
    """
    def bboo_metaheuristic(budget, dim):
        return BBOBMetaheuristic(budget, dim)(func)
    def func_bboo_metaheuristic(x):
        return bboo_metaheuristic(x)
    def func_bboo_metaheuristic_refined(x):
        new_x = x.copy()
        if np.random.rand() < prob_change:
            new_x = np.random.uniform(-5.0, 5.0, (dim,))
        return func_bboo_metaheuristic_refined(new_x)
    return BBOBMetaheuristic(budget, dim)(func_bboo_metaheuristic_refined)

# Description: BBOB Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize
#
# def bboo_metaheuristic(func, budget, dim, prob_change):
#     return BBOBMetaheuristic(budget, dim)(func)
#
# def func(x):
#     return x[0]**2 + x[1]**2
#
# budget = 1000
# dim = 2
# prob_change = 0.2
# metaheuristic = bboo_metaheuristic(func, budget, dim, prob_change)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')

# Example usage:
# refined_metaheuristic = bboo_metaheuristic_refined(func, budget, dim, prob_change)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)], n_savings=10)
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')