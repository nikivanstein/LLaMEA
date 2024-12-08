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
    def update_individual(self, individual, logger):
        """
        Updates the individual's strategy by refining its search space.

        Parameters:
        ----------
        individual : Individual
            The current individual to update.
        logger : Logger
            The logger object used to track the progress of the optimization process.
        """
        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Update the bounds based on the individual's strategy
        for dim in range(self.dim):
            if individual.strategies[dim] == "random":
                bounds[dim] = np.random.uniform(-5.0, 5.0)
            elif individual.strategies[dim] == "gradient":
                bounds[dim] = np.array([np.sqrt(5.0 - dim), np.sqrt(dim)])

        # Update the individual's parameters
        individual.params = np.random.uniform(bounds[0, 0], bounds[1, 0], self.dim)

        # Update the individual's fitness
        individual.f(individual, logger)