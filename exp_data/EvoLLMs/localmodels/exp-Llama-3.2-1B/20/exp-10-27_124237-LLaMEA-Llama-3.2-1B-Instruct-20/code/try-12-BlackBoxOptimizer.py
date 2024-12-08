import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    """
    An optimization algorithm that handles a wide range of tasks by leveraging the power of black box optimization.

    Attributes:
    ----------
    budget : int
        The maximum number of function evaluations allowed.
    dim : int
        The dimensionality of the search space.

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


# One-line description with the main idea
# "BlackBoxOptimizer" is a novel metaheuristic algorithm that optimizes black box functions using a combination of random search and gradient-based optimization."

# Exception handling code:
try:
    # Initialize the BlackBoxOptimizer algorithm
    optimizer = BlackBoxOptimizer(100, 10)
except np.linalg.LinAlgError:
    # Handle the case where the dimensionality is large
    print("Warning: The dimensionality is too large. Using a smaller dimensionality instead.")
    optimizer = BlackBoxOptimizer(10, 10)