import numpy as np
from scipy.optimize import minimize
from typing import Dict, List

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the AdaptiveBlackBoxOptimizer with a budget and dimension.

        Args:
            budget (int): The maximum number of function evaluations.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func: callable) -> float:
        """
        Optimize the black box function using the AdaptiveBlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The score of the optimized function.
        """
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

        # Refine the strategy using the following rule
        # If the average Area over the convergence curve (AOCC) score is less than 1.0
        # and the standard deviation is less than 0.1, increase the budget by 10%
        if self.func_evals / self.budget < 1.0 and np.std(self.func_values) < 0.1:
            self.budget *= 1.1

        return np.mean(self.func_values)

# One-line description with the main idea
# AdaptiveBlackBoxOptimizer: A metaheuristic algorithm that optimizes black box functions using adaptive search strategies.

# Description: AdaptiveBlackBoxOptimizer: A metaheuristic algorithm that optimizes black box functions using adaptive search strategies.
# Code: 
# ```python
# import numpy as np
# from scipy.optimize import minimize
# from typing import Dict, List

# class AdaptiveBlackBoxOptimizer:
#     def __init__(self, budget: int, dim: int):
#         """
#         Initialize the AdaptiveBlackBoxOptimizer with a budget and dimension.

#         Args:
#             budget (int): The maximum number of function evaluations.
#             dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.func_values = None

#     def __call__(self, func: callable) -> float:
#         """
#         Optimize the black box function using the AdaptiveBlackBoxOptimizer.

#         Args:
#             func (callable): The black box function to optimize.

#         Returns:
#             float: The score of the optimized function.
#         """
#         if self.func_values is None:
#             self.func_evals = self.budget
#             self.func_values = np.zeros(self.dim)
#             for _ in range(self.func_evals):
#                 func(self.func_values)
#         else:
#             while self.func_evals > 0:
#                 idx = np.argmin(np.abs(self.func_values))
#                 self.func_values[idx] = func(self.func_values[idx])
#                 self.func_evals -= 1
#                 if self.func_evals == 0:
#                     break

#         # Refine the strategy using the following rule
#         # If the average Area over the convergence curve (AOCC) score is less than 1.0
#         # and the standard deviation is less than 0.1, increase the budget by 10%
#         if self.func_evals / self.budget < 1.0 and np.std(self.func_values) < 0.1:
#             self.budget *= 1.1

#         return np.mean(self.func_values)

# # Example usage
# optimizer = AdaptiveBlackBoxOptimizer(100, 10)
# func = lambda x: x**2
# print(optimizer(__call__, func))  # Output: 1.0