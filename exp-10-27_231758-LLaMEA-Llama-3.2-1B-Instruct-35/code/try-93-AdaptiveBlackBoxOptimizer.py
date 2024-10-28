import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
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

    def __str__(self):
        return f"AdaptiveBlackBoxOptimizer: Optimizes {self.dim}-dimensional black box function with {self.budget} evaluations"

    def adapt_strategy(self, func, budget, dim, noise, c1, c2, alpha, beta):
        """
        Adapt the strategy of the AdaptiveBlackBoxOptimizer algorithm.
        
        Parameters:
        func (callable): The black box function to optimize.
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        noise (float): The standard deviation of the noise added to the function values.
        c1 (float): The coefficient of the first term in the C2 strategy.
        c2 (float): The coefficient of the second term in the C2 strategy.
        alpha (float): The learning rate for the C2 strategy.
        beta (float): The regularization parameter for the C2 strategy.
        
        Returns:
        dict: A dictionary containing the updated strategy parameters.
        """
        # Initialize the strategy parameters
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta

        # Refine the strategy based on the individual lines
        self.c1 *= 0.7  # Reduce the contribution of the first term
        self.c2 *= 0.7  # Reduce the contribution of the second term
        self.alpha *= 0.8  # Increase the learning rate
        self.beta *= 0.8  # Increase the regularization parameter

        return {
            'c1': self.c1,
            'c2': self.c2,
            'alpha': self.alpha,
            'beta': self.beta
        }

# Description: Refine the AdaptiveBlackBoxOptimizer algorithm using the C2 strategy.
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer: Optimizes 3-dimensional black box function with 1000 evaluations
# 
# Parameters:
#   func (callable): The black box function to optimize.
#   budget (int): The number of function evaluations allowed.
#   dim (int): The dimensionality of the search space.
#   noise (float): The standard deviation of the noise added to the function values.
#   c1 (float): The coefficient of the first term in the C2 strategy.
#   c2 (float): The coefficient of the second term in the C2 strategy.
#   alpha (float): The learning rate for the C2 strategy.
#   beta (float): The regularization parameter for the C2 strategy.
# 
# Returns:
#   dict: A dictionary containing the updated strategy parameters.