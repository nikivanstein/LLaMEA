import numpy as np
from scipy.optimize import minimize, differential_evolution

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

    def hybrid_search(self, func, bounds, num_evals):
        """
        A hybrid search algorithm that combines random search with gradient-based optimization.

        Parameters:
        ----------
        func : function
            The black box function to optimize.
        bounds : list
            A list of bounds for the search space.
        num_evals : int
            The maximum number of function evaluations allowed.

        Returns:
        -------
        tuple
            A tuple containing the optimized parameters and the optimized function value.
        """
        # Initialize the population with random values within the bounds
        population = np.random.uniform(-5.0, 5.0, (num_evals, self.dim))

        # Define the function to minimize (in this case, the negative of the function value)
        def neg_func(params):
            return -func(params)

        # Use the differential evolution algorithm to optimize the function
        result = differential_evolution(neg_func, bounds, popsize=100, num_evals=num_evals)

        # Return the optimized parameters and the optimized function value
        return result.x, -result.fun


# One-line description with the main idea
# "HybridBlackBoxOptimizer" combines random search with gradient-based optimization to improve the convergence rate and efficiency of black box optimization."