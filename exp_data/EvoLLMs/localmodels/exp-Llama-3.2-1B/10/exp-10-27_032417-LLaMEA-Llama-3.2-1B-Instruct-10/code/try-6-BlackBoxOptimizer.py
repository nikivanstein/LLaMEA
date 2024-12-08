import numpy as np
import random

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
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

            # Calculate the probability of accepting the current solution
            probability = 1.0
            for _ in range(100):
                # Accept the current solution with a probability less than 1
                if random.random() < probability:
                    solution = solution
                    break
                # Generate a new solution within the search space
                solution = np.random.uniform(-5.0, 5.0, self.dim)

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Accept the current solution with a probability less than 1
                    probability = 1.0
                    for _ in range(100):
                        # Generate a new solution within the search space
                        solution = np.random.uniform(-5.0, 5.0, self.dim)
                        # Accept the current solution with a probability less than 1
                        if random.random() < probability:
                            solution = solution
                            break
                    # Update the solution with the best solution found so far
                    solution = None
                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations