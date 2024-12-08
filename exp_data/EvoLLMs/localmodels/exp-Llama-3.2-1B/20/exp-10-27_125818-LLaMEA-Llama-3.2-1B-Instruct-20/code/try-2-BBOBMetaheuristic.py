# Description: Adaptive BBOB Metaheuristic
# Code: 
# ```python
import numpy as np
import random
import time

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
        self.logger = None
        self.new_individuals = None
        self.best_individuals = None

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
            self.logger = time.time()
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
                    # Record the start time
                    start_time = time.time()
                # Change the strategy
                if random.random() < 0.2:
                    # Use the new individual's fitness as the new fitness
                    self.f = self.func(self.x)
                else:
                    # Use the current individual's fitness as the new fitness
                    self.f = self.f
                # Check if the new point is better than the current point
                if self.f < self.f + 1e-6:  # add a small value to avoid division by zero
                    # Update the current point
                    self.x = self.x
                    self.f = self.f
                    # Record the end time
                    end_time = time.time()
                    # Update the best individual
                    if end_time - start_time < 0.1:
                        self.best_individual = self.x
                    # Update the new individuals
                    if end_time - start_time < 0.1:
                        self.new_individuals = [self.x]
                    else:
                        self.new_individuals.append(self.x)
                    # Update the best individuals
                    if end_time - start_time < 0.1:
                        self.best_individuals = self.x
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
# new_individuals = []
# best_individuals = None
# for _ in range(10):
#     res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
#     x0 = res.x
#     new_individuals.append(x0)
#     if res.fun < best_individuals:
#         best_individuals = res.x
#     x0 = new_individuals[-1]
# x0 = best_individuals
# print(f'Optimized function: {best_individuals}')
# print(f'Optimized parameters: {best_individuals}')