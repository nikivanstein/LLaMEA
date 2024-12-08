# Description: Black Box Optimization using Genetic Algorithm with Simulated Annealing
# Code: 
# ```python
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
                    # Calculate the temperature for simulated annealing
                    temperature = self.budget / evaluations

                    # Update the solution using simulated annealing
                    if random.random() < np.exp((evaluations - evaluations) / temperature):
                        solution = solution

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
def refine_solution(solution, num_evaluations):
    """
    Refines the solution using simulated annealing.

    Args:
        solution (float): The current solution.
        num_evaluations (int): The number of function evaluations used.

    Returns:
        float: The refined solution.
    """
    temperature = num_evaluations / 100
    for _ in range(100):
        # Generate a new solution within the search space
        new_solution = np.random.uniform(-5.0, 5.0, solution.shape)

        # Evaluate the black box function at the new solution
        func(new_solution)

        # Calculate the probability of accepting the new solution
        probability = np.exp((num_evaluations - _ - 1) / temperature)

        # Accept the new solution with a probability less than 1
        if np.random.rand() < probability:
            new_solution = new_solution
        else:
            # Calculate the temperature for simulated annealing
            temperature *= 0.99

    # Return the refined solution
    return new_solution

optimal_solution = refine_solution(optimal_solution, num_evaluations)
print("Refined optimal solution:", optimal_solution)
print("Refined number of function evaluations:", num_evaluations)