import numpy as np
from scipy.optimize import minimize

class AdaptiveBlackBoxOptimizer:
    """
    An optimization algorithm that handles a wide range of tasks by leveraging the power of black box optimization.

    Attributes:
    ----------
    budget : int
        The maximum number of function evaluations allowed.
    dim : int
        The dimensionality of the search space.
    adaptive_search : bool
        Whether to use adaptive search or not.
    adaptive_budget : int
        The maximum number of function evaluations allowed in adaptive search.

    Methods:
    -------
    __init__(self, budget, dim, adaptive_search=True, adaptive_budget=100)
        Initializes the optimization algorithm with the given budget and dimensionality.
    def __call__(self, func)
        Optimizes the black box function `func` using `self.budget` function evaluations.
    """

    def __init__(self, budget, dim, adaptive_search=True, adaptive_budget=100):
        """
        Initializes the optimization algorithm with the given budget and dimensionality.

        Parameters:
        ----------
        budget : int
            The maximum number of function evaluations allowed.
        dim : int
            The dimensionality of the search space.
        adaptive_search : bool
            Whether to use adaptive search or not. Defaults to True.
        adaptive_budget : int
            The maximum number of function evaluations allowed in adaptive search. Defaults to 100.
        """
        self.budget = budget
        self.dim = dim
        self.adaptive_search = adaptive_search
        self.adaptive_budget = adaptive_budget

    def __call__(self, func):
        """
        Optimizes the black box function `func` using `self.budget` function evaluations.

        Parameters:
        ----------
        func : function
            The black box function to optimize.

        Returns:
        -------
        tuple
            A tuple containing the optimized parameters and the optimized function value.
        """
        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Initialize the parameters with random values within the bounds
        params = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the function to minimize (in this case, the negative of the function value)
        def neg_func(params):
            return -func(params)

        # Use the minimize function from SciPy to optimize the function
        if self.adaptive_search:
            # Use adaptive search to refine the search space
            result = minimize(neg_func, params, method="SLSQP", bounds=bounds, options={"maxiter": self.adaptive_budget})
            return result.x, -result.fun
        else:
            # Use standard search to optimize the function
            result = minimize(neg_func, params, method="SLSQP", bounds=bounds, options={"maxiter": self.budget})
            return result.x, -result.fun


# One-line description with the main idea
# "AdaptiveBlackBoxOptimizer" is a novel metaheuristic algorithm that optimizes black box functions using adaptive search and gradient-based optimization."