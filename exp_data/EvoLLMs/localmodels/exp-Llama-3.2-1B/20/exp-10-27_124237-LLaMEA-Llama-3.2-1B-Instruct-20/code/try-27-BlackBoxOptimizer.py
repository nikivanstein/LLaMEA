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

# Refining the strategy
def BlackBoxOptimizer_with_refined_strategy(self, func, bounds, initial_params):
    """
    A refined version of the BlackBoxOptimizer algorithm that uses a more sophisticated strategy.

    Parameters:
    ----------
    func : function
        The black box function to optimize.
    bounds : list
        A list of tuples representing the lower and upper bounds for each dimension.
    initial_params : list
        A list of initial values for the parameters.

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

    # Use a more sophisticated strategy, such as a genetic algorithm or a simulated annealing algorithm
    # For this example, we will use a simple strategy that randomly perturbs the parameters
    for _ in range(self.budget):
        # Randomly perturb the parameters
        perturbed_params = params + np.random.normal(0, 1, self.dim)

        # Evaluate the function at the perturbed parameters
        new_func_value = func(perturbed_params)

        # If the new function value is better than the current best function value, update the parameters
        if new_func_value < -func(params):
            params = perturbed_params

    # Return the optimized parameters and the optimized function value
    return params, -func(params)


# Example usage
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return x[0]**2 + x[1]**2

    # Define the bounds for the search space
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    # Initialize the optimization algorithm
    optimizer = BlackBoxOptimizer(100, 2)

    # Optimize the black box function
    optimized_params, optimized_func_value = optimizer(func, bounds)

    # Print the optimized parameters and the optimized function value
    print("Optimized parameters:", optimized_params)
    print("Optimized function value:", optimized_func_value)