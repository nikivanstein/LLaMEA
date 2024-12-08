import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import random

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
    def select_strategy(self, individual, logger):
        """
        Selects a strategy for the individual based on its fitness value and the budget.

        Parameters:
        ----------
        individual : Individual
            The current individual to be evaluated.
        logger : Logger
            The logger object used to track the optimization process.

        Returns:
        -------
        tuple
            A tuple containing the selected strategy and the updated individual.
        """
        # Calculate the fitness value of the individual
        fitness = individual.f

        # If the fitness value is less than 0.2, use a random search strategy
        if fitness < 0.2:
            new_individual = individual
            while True:
                # Generate a new individual using a linear interpolation of the current individual
                new_individual = individual + (0.5 * (individual.x - individual.x0) * np.linspace(0, 1, individual.dim))

                # Evaluate the fitness value of the new individual
                new_fitness = new_individual.f

                # If the fitness value is less than 0.2, select the new individual
                if new_fitness < 0.2:
                    new_individual = new_individual
                    break

                # Otherwise, select the current individual
                else:
                    new_individual = individual
        # If the fitness value is greater than or equal to 0.2, use a gradient-based optimization strategy
        else:
            # Calculate the gradient of the fitness function at the current individual
            gradient = expit(individual.f)

            # Initialize the direction of the search
            direction = [0.0] * individual.dim

            # Calculate the step size
            step_size = 0.1

            # Iterate until the budget is reached
            for _ in range(self.budget):
                # Evaluate the fitness value of the current individual plus the step size in the direction
                new_fitness = individual.f + step_size * np.dot(gradient, direction)

                # If the fitness value is less than the current fitness value, select the new individual
                if new_fitness < fitness:
                    new_individual = individual + direction
                    break

                # Otherwise, select the current individual
                else:
                    new_individual = individual

            # Return the selected strategy and the updated individual
            return "random", new_individual

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