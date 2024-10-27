import numpy as np
from scipy.optimize import minimize

class MetaHeuristicBlackBoxOptimizer:
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
    def update_individual(self, individual, logger):
        """
        Updates the individual's strategy by refining its search space.

        Parameters:
        ----------
        individual : Individual
            The current individual to update.
        logger : Logger
            The logger to use for logging updates.

        Returns:
        -------
        Individual
            The updated individual.
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

        # Update the individual's parameters
        new_individual = individual.__call__(func, result.x, result.fun, logger)

        return new_individual