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
                probability = 0.1  # Change this value to change the individual's strategy

                # Accept the current solution with a probability less than 1
                if np.random.rand() < probability:
                    solution = solution
                else:
                    # Generate a new individual with a slightly different strategy
                    new_individual = self.generate_new_individual(solution)

                    # Evaluate the new individual
                    new_evaluations = 0
                    while new_evaluations < self.budget:
                        new_individual = np.random.uniform(-5.0, 5.0, self.dim)
                        new_evaluations += 1
                        func(new_individual)

                    # Update the solution with the new individual
                    solution = new_individual

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


    def generate_new_individual(self, solution):
        """
        Generates a new individual with a slightly different strategy.

        Args:
            solution (numpy array): The current solution.

        Returns:
            numpy array: The new individual.
        """
        # Generate a new individual with a slightly different strategy
        new_individual = solution.copy()
        for i in range(self.dim):
            # Change the strategy of the individual with a small probability
            if np.random.rand() < 0.1:
                new_individual[i] = random.uniform(-5.0, 5.0)
        return new_individual


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)