import numpy as np
from scipy.optimize import minimize, differential_evolution

class AdaptiveBlackBoxOptimizer:
    """
    An adaptive optimization algorithm that handles a wide range of tasks by leveraging the power of black box optimization.

    Attributes:
    ----------
    budget : int
        The maximum number of function evaluations allowed.
    dim : int
        The dimensionality of the search space.
    line_search_multiplier : float
        The multiplier used in the adaptive line search.
    adaptive_line_search : bool
        Whether to use adaptive line search.
    bounds : list
        The bounds for the search space.

    Methods:
    -------
    __init__(self, budget, dim)
        Initializes the optimization algorithm with the given budget and dimensionality.
    def __call__(self, func)
        Optimizes the black box function `func` using `self.budget` function evaluations.
    """

    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm with the given budget and dimensionality.

        Parameters:
        ----------
        budget : int
            The maximum number of function evaluations allowed.
        dim : int
            The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.line_search_multiplier = 0.2
        self.adaptive_line_search = False
        self.bounds = [(-5.0, 5.0) for _ in range(self.dim)]

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
        result = minimize(neg_func, params, method="SLSQP", bounds=bounds, options={"maxiter": self.budget})

        # Return the optimized parameters and the optimized function value
        return result.x, -result.fun

    def adaptive_line_search(self, func, params, func_evaluations):
        """
        Performs adaptive line search using the provided function evaluations.

        Parameters:
        ----------
        func : function
            The black box function to optimize.
        params : array
            The current parameters.
        func_evaluations : int
            The number of function evaluations to perform.

        Returns:
        -------
        array
            The optimized parameters.
        """
        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Initialize the parameters with random values within the bounds
        params = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the function to minimize (in this case, the negative of the function value)
        def neg_func(params):
            return -func(params)

        # Use the minimize function from SciPy to optimize the function
        result = minimize(neg_func, params, method="SLSQP", bounds=bounds, options={"maxiter": func_evaluations})

        # Perform adaptive line search
        if self.adaptive_line_search:
            # Update the parameters using the adaptive line search
            params = np.array([self.line_search_multiplier * p + (1 - self.line_search_multiplier) * np.sign(result.x) for p in params])

        # Return the optimized parameters
        return params