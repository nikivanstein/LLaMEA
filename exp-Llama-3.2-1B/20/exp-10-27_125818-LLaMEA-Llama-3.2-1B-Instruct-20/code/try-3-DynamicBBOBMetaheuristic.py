import numpy as np
import random
from scipy.optimize import minimize

class DynamicBBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the DynamicBBOBMetaheuristic with a given budget and dimensionality.

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
                    # Refine the strategy by changing the individual
                    if random.random() < 0.2:
                        self.x = self.x[0] + random.uniform(-0.1, 0.1)
                        self.f = self.func(self.x)
                    elif random.random() < 0.4:
                        self.x = self.x[1] + random.uniform(-0.1, 0.1)
                        self.f = self.func(self.x)
                    else:
                        self.x = self.x[0] + random.uniform(-0.1, 0.1)
                        self.f = self.func(self.x)
                # Decrease the budget
                self.budget -= 1
            # Return the optimized function value
            return self.f

# Description: Dynamic BBOB Metaheuristic
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize

# def dynamic_bboo_metaheuristic(func, budget, dim):
#     return DynamicBBOBMetaheuristic(budget, dim)(func)

# def func(x):
#     return x[0]**2 + x[1]**2

# budget = 1000
# dim = 2
# metaheuristic = dynamic_bboo_metaheuristic(func, budget, dim)
# x0 = [1.0, 1.0]
# res = minimize(func, x0, method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)])
# print(f'Optimized function: {res.fun}')
# print(f'Optimized parameters: {res.x}')