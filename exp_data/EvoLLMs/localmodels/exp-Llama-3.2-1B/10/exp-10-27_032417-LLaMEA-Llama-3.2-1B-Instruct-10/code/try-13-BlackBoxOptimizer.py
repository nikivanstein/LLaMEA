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

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = solution
                else:
                    # Update the solution with the best solution found so far
                    solution = None

            # Refine the solution using simulated annealing
            if evaluations > 0 and evaluations < self.budget:
                # Generate a new solution with a probability based on the current temperature
                new_solution = solution + np.random.uniform(-1.0, 1.0, self.dim)

                # Accept the new solution with a probability less than 1
                if np.random.rand() < np.exp((evaluations - evaluations) / self.budget):
                    solution = new_solution

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Refine the solution using simulated annealing
for _ in range(10):
    new_solution = optimizer.func(optimal_solution)
    optimizer.func(new_solution)
    print("Refined solution:", new_solution)
    print("Number of function evaluations:", num_evaluations)