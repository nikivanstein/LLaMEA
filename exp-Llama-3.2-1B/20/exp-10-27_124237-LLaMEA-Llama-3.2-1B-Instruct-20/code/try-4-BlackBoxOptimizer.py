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
    def mutate(self, individual):
        """
        Applies adaptive mutation strategies to the individual to refine its strategy.

        Parameters:
        ----------
        individual : Individual
            The individual to mutate.

        Returns:
        -------
        Individual
            The mutated individual.
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
        # Initialize the population with random individuals
        population = [self.evaluate_fitness(individual) for individual in self.generate_population(self.budget)]

        # Define the mutation rate
        mutation_rate = 0.2

        # Apply adaptive mutation strategies to the population
        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population.index(max(population))

            # Mutate the fittest individual
            mutated_individual = self.mutate(population[fittest_individual])

            # Replace the fittest individual with the mutated individual
            population[fittest_individual] = mutated_individual

            # Remove the fittest individual from the population
            population.remove(population[fittest_individual])

        # Return the optimized function value
        return population[0], -population[0]


# One-line description with the main idea
# "BlackBoxOptimizer" is a novel metaheuristic algorithm that optimizes black box functions using a combination of random search, gradient-based optimization, and adaptive mutation strategies."