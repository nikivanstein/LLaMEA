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
                if self.f < self.f + 1e-6:  # add a small value to avoid division by zero
                    # Update the current point
                    self.x = self.x
                    self.f = self.f
                # Update the iterations
                self.iterations += 1
                # If the iterations exceeds the budget, refine the strategy
                if self.iterations >= self.budget:
                    self.space = self.space + np.random.uniform(-0.1, 0.1, (self.dim,))
                    self.x = self.x + np.random.uniform(-0.1, 0.1, (self.dim,))
                    self.f = self.func(self.x)
            # Return the optimized function value
            return self.f

# Description: BBOB Metaheuristic with Adaptive Search Strategy
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

# BBOB Metaheuristic with Adaptive Search Strategy
# Description: BBOB Metaheuristic with Adaptive Search Strategy
# Code: 
# ```python
# import numpy as np

class BBOBMetaheuristicWithAdaptiveSearch:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBMetaheuristicWithAdaptiveSearch with a given budget and dimensionality.

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
        self.adaptive_strategy = False

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
                # Update the iterations
                self.iterations += 1
                # If the iterations exceeds the budget, refine the strategy
                if self.iterations >= self.budget:
                    if self.adaptive_strategy:
                        # Refine the search space using the adaptive strategy
                        self.space = self.space + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.x = self.x + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.f = self.func(self.x)
                    # Return the optimized function value
                    return self.f
                # Update the adaptive strategy
                if self.iterations % 100 == 0:
                    self.adaptive_strategy = not self.adaptive_strategy
            # Return the optimized function value
            return self.f

# Description: BBOB Metaheuristic with Adaptive Search Strategy
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

# BBOB Metaheuristic with Adaptive Search Strategy
# Description: BBOB Metaheuristic with Adaptive Search Strategy
# Code: 
# ```python
# import numpy as np

class BBOBMetaheuristicWithAdaptiveSearchWithMultiStrategy:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBMetaheuristicWithAdaptiveSearchWithMultiStrategy with a given budget and dimensionality.

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
        self.multi_strategies = []
        self.adaptive_strategy = False

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
                # Update the iterations
                self.iterations += 1
                # If the iterations exceeds the budget, refine the strategy
                if self.iterations >= self.budget:
                    if self.adaptive_strategy:
                        # Refine the search space using the adaptive strategy
                        self.space = self.space + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.x = self.x + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.f = self.func(self.x)
                    # Update the multi-strategies
                    for strategy in self.multi_strategies:
                        # Refine the search space using the multi-strategy
                        self.space = strategy.space + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.x = strategy.x + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.f = strategy.f
                    # Return the optimized function value
                    return self.f
                # Update the adaptive strategy
                if self.iterations % 100 == 0:
                    self.adaptive_strategy = not self.adaptive_strategy
            # Return the optimized function value
            return self.f

# Description: BBOB Metaheuristic with Adaptive Search Strategy
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

# BBOB Metaheuristic with Adaptive Search Strategy
# Description: BBOB Metaheuristic with Adaptive Search Strategy
# Code: 
# ```python
# import numpy as np

class BBOBMetaheuristicWithAdaptiveSearchMultiStrategy:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBMetaheuristicWithAdaptiveSearchMultiStrategy with a given budget and dimensionality.

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
        self.multi_strategies = []
        self.adaptive_strategy = False

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
                # Update the iterations
                self.iterations += 1
                # If the iterations exceeds the budget, refine the strategy
                if self.iterations >= self.budget:
                    # Update the multi-strategies
                    for strategy in self.multi_strategies:
                        # Refine the search space using the multi-strategy
                        self.space = strategy.space + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.x = strategy.x + np.random.uniform(-0.1, 0.1, (self.dim,))
                        self.f = strategy.f
                    # Update the adaptive strategy
                    if self.iterations % 100 == 0:
                        self.adaptive_strategy = not self.adaptive_strategy
                    # Return the optimized function value
                    return self.f
                # Update the adaptive strategy
                if self.iterations % 100 == 0:
                    self.adaptive_strategy = not self.adaptive_strategy
            # Return the optimized function value
            return self.f

# Description: BBOB Metaheuristic with Adaptive Search Strategy
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

# BBOB Metaheuristic with Adaptive Search Strategy
# Description: BBOB Metaheuristic with Adaptive Search Strategy
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

# BBOB Metaheuristic with Adaptive Search Strategy
# Description: BBOB Metaheuristic with Adaptive Search Strategy
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

# BBOB Metaheuristic with Adaptive Search Strategy
# Description: BBOB Metaheuristic with Adaptive Search Strategy
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