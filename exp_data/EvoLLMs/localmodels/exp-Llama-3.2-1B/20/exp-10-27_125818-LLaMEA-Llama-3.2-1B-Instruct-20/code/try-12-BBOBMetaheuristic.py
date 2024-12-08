import numpy as np
import random
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

# Novel Heuristic Algorithm: Adaptive Resampling with Evolutionary Strategies
class AdaptiveResamplingMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        """
        Initialize the Adaptive Resampling Metaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        """
        super().__init__(budget, dim)
        self.evolutionary_strategy = None

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
            # Resample the points using evolutionary strategies
            if self.evolutionary_strategy:
                self.evolutionary_strategy.update(self.x, self.f)
            # Return the optimized function value
            return self.f

    def __str__(self):
        return f"Adaptive Resampling Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems."

# Description: Adaptive Resampling Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
# import numpy as np
# import random
# import copy
# import numpy as np
# import matplotlib.pyplot as plt
#
# class EvolutionaryStrategy:
#     def __init__(self, func, bounds):
#         self.func = func
#         self.bounds = bounds
#
#     def update(self, x, f):
#         # Sample a new point in the search space
#         new_x = np.random.uniform(self.bounds[0], self.bounds[1], (len(x), self.dim))
#         # Evaluate the function at the new point
#         new_f = self.func(new_x)
#         # Check if the new point is better than the current point
#         if new_f < f:
#             # Update the current point
#             x = new_x
#             f = new_f
#         return x, f
#
# def bboo_metaheuristic(func, budget, dim):
#     return AdaptiveResamplingMetaheuristic(budget, dim)(func)
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

# One-line description: Adaptive Resampling Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
# import numpy as np
# import random
# import copy
# import numpy as np
# import matplotlib.pyplot as plt
#
# class EvolutionaryStrategy:
#     def __init__(self, func, bounds):
#         self.func = func
#         self.bounds = bounds
#
#     def update(self, x, f):
#         # Sample a new point in the search space
#         new_x = copy.deepcopy(x)
#         # Evaluate the function at the new point
#         new_f = self.func(new_x)
#         # Check if the new point is better than the current point
#         if new_f < f:
#             # Update the current point
#             new_x = [new_f, new_f]
#             new_f = new_f
#         return new_x, new_f
#
# def bboo_metaheuristic(func, budget, dim):
#     return AdaptiveResamplingMetaheuristic(budget, dim)(func)
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

# One-line description: Adaptive Resampling Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
# import numpy as np
# import random
# import copy
# import numpy as np
# import matplotlib.pyplot as plt
#
# class EvolutionaryStrategy:
#     def __init__(self, func, bounds):
#         self.func = func
#         self.bounds = bounds
#
#     def update(self, x, f):
#         # Sample a new point in the search space
#         new_x = copy.deepcopy(x)
#         # Evaluate the function at the new point
#         new_f = self.func(new_x)
#         # Check if the new point is better than the current point
#         if new_f < f:
#             # Update the current point
#             new_x = [new_f, new_f]
#             new_f = new_f
#         return new_x, new_f
#
# def bboo_metaheuristic(func, budget, dim):
#     return AdaptiveResamplingMetaheuristic(budget, dim)(func)
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

# One-line description: Adaptive Resampling Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
# import numpy as np
# import random
# import copy
# import numpy as np
# import matplotlib.pyplot as plt
#
# class EvolutionaryStrategy:
#     def __init__(self, func, bounds):
#         self.func = func
#         self.bounds = bounds
#
#     def update(self, x, f):
#         # Sample a new point in the search space
#         new_x = copy.deepcopy(x)
#         # Evaluate the function at the new point
#         new_f = self.func(new_x)
#         # Check if the new point is better than the current point
#         if new_f < f:
#             # Update the current point
#             new_x = [new_f, new_f]
#             new_f = new_f
#         return new_x, new_f
#
# def bboo_metaheuristic(func, budget, dim):
#     return AdaptiveResamplingMetaheuristic(budget, dim)(func)
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