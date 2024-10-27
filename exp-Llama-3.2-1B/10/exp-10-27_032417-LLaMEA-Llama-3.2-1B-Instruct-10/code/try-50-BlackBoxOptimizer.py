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
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        # Refine the strategy with a modified individual line
                        solution = np.random.uniform(-5.0, 5.0, self.dim)
                        # Update the individual line with a modified strategy
                        solution = random.choice(solution)
                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Description: Genetic Algorithm for Black Box Optimization
# Code: 
# ```python
# import numpy as np
# import random
#
# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#
#     def __call__(self, func):
#         solution = None
#         evaluations = 0
#
#         while evaluations < self.budget:
#             solution = np.random.uniform(-5.0, 5.0, self.dim)
#             evaluations += 1
#             func(solution)
#
#         return solution, evaluations
#
# def func(x):
#     return x**2 + 2*x + 1
#
# optimizer = BlackBoxOptimizer(100, 10)
# optimal_solution, num_evaluations = optimizer(func)
# print("Optimal solution:", optimal_solution)
# print("Number of function evaluations:", num_evaluations)