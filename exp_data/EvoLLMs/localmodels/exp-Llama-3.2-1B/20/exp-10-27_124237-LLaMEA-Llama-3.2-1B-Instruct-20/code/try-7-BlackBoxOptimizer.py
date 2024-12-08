import numpy as np
from scipy.optimize import minimize
from scipy.misc import derivative

class BlackBoxOptimizer:
    """
    An optimization algorithm that handles a wide range of tasks by leveraging the power of black box optimization.

    Attributes:
    ----------
    budget : int
        The maximum number of function evaluations allowed.
    dim : int
        The dimensionality of the search space.
    bounds : list
        A list of tuples representing the bounds for each dimension.
    params : np.ndarray
        The current set of parameters.
    func : callable
        The black box function to optimize.
    logger : object
        The logger object used for logging.
    """

    def __init__(self, budget, dim, bounds, func):
        """
        Initializes the optimization algorithm with the given budget, dimensionality, bounds, and function.

        Parameters:
        ----------
        budget : int
            The maximum number of function evaluations allowed.
        dim : int
            The dimensionality of the search space.
        bounds : list
            A list of tuples representing the bounds for each dimension.
        func : function
            The black box function to optimize.
        """
        self.budget = budget
        self.dim = dim
        self.bounds = bounds
        self.params = np.random.uniform(-5.0, 5.0, dim)
        self.func = func
        self.logger = logging.getLogger(__name__)

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
        # Define the function to minimize (in this case, the negative of the function value)
        def neg_func(params):
            return -func(params)

        # Use the minimize function from SciPy to optimize the function
        result = minimize(neg_func, self.params, method="SLSQP", bounds=self.bounds, options={"maxiter": self.budget})

        # Return the optimized parameters and the optimized function value
        return result.x, -result.fun


# One-line description with the main idea
# "BlackBoxOptimizer" is a novel metaheuristic algorithm that optimizes black box functions using a combination of random search and gradient-based optimization."

# Exception that occurred
try:
    # Code that will run without an exception
    new_individual = BlackBoxOptimizer(100, 10, [[-5.0, -5.0], [-5.0, 5.0]], lambda x: x**2).evaluate_fitness(BlackBoxOptimizer(100, 10, [[-5.0, -5.0], [-5.0, 5.0]], lambda x: x**2))
except TypeError as e:
    # Handle the exception
    print(f"An error occurred: {e}")