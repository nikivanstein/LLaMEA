import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, initial_individual, logger):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            initial_individual (list): The initial individual.
            logger (object): The logger object.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Initialize the population with the initial individual
        population = [initial_individual.copy()]

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a new individual using mutation and crossover
            new_individual = self.mutate(population[-1], self.logger)

            # Evaluate the function at the current individual
            value = func(new_individual)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = new_individual

            # Add the new individual to the population
            population.append(new_individual)

        # Return the optimized value
        return best_value

    def mutate(self, individual, logger):
        """
        Mutate the individual using a simple crossover strategy.

        Args:
            individual (list): The individual to mutate.
            logger (object): The logger object.

        Returns:
            list: The mutated individual.
        """
        # Randomly select two points in the search space
        point1 = random.randint(0, self.dim - 1)
        point2 = random.randint(0, self.dim - 1)

        # Create a new individual by combining the two points
        new_individual = individual[:point1] + [individual[point2]] + individual[point1 + 1:]

        # Add the new individual to the population
        population.append(new_individual)

        # Update the logger
        logger.update("Mutation", "New individual: ", new_individual)

        return new_individual

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABA)
# Code: 