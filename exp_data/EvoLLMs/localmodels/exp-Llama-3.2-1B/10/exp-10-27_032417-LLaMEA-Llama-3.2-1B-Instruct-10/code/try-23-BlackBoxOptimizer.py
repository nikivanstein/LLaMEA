import numpy as np
import random

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.

    The algorithm uses a combination of genetic algorithm and simulated annealing to find the optimal solution.
    """

    def __init__(self, budget, dim):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimizes a black box function using the optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the solution and the number of function evaluations
        solution = None
        evaluations = 0

        # Iterate over the range of possible solutions
        while evaluations < self.budget:
            # Generate a random solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the black box function at the current solution
            evaluations += 1
            func(solution)

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                # Calculate the probability of accepting the current solution
                probability = np.exp((evaluations - evaluations) / self.budget)

                # Accept the current solution with a probability less than 1
                if np.random.rand() < probability:
                    solution = solution
                else:
                    # Refine the solution using a new line of code
                    new_individual = self.evaluate_fitness(solution)
                    solution = new_individual

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


def evaluate_fitness(individual, logger):
    """
    Evaluates the fitness of a given individual.

    Args:
        individual (numpy array): The individual to evaluate.
        logger (object): The logger to use for tracking the fitness.

    Returns:
        float: The fitness of the individual.
    """
    # Simulate the function evaluations
    for i in range(self.dim):
        individual[i] = func(individual[i])

    # Calculate the fitness as the negative of the sum of the function evaluations
    fitness = -np.sum(individual)

    # Update the logger with the fitness
    logger.update_fitness(fitness)

    return fitness


def func(x):
    """
    The black box function to optimize.

    Args:
        x (numpy array): The input to the function.

    Returns:
        float: The output of the function.
    """
    return x**2 + 2*x + 1


# Example usage:
optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)