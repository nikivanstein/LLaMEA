import numpy as np
import random
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
    def __str__(self):
        return f"BlackBoxOptimizer: (Score: {self.score}, Strategy: Genetic Programming)"

    def __str__(self):
        return f"BlackBoxOptimizer: (Score: {self.score}, Strategy: Genetic Programming)"

    def __repr__(self):
        return f"BlackBoxOptimizer(budget={self.budget}, dim={self.dim})"

# One-line description with the main idea
# "BlackBoxOptimizer" is a novel metaheuristic algorithm that optimizes black box functions using a combination of random search, gradient-based optimization, and genetic programming."