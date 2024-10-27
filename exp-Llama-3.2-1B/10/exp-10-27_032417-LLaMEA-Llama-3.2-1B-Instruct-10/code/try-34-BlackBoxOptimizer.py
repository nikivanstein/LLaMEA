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

                    # Update the probability of accepting the solution based on the current solution
                    if evaluations > 0:
                        # Calculate the probability of accepting the solution based on the current solution
                        probability = np.exp((evaluations - 1) / self.budget)

                        # Accept the solution with a probability less than 1
                        if np.random.rand() < probability:
                            solution = solution

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

# Refine the solution based on the probability of acceptance
def refine_solution(solution, probability):
    """
    Refines the solution based on the probability of acceptance.

    Args:
        solution (float): The current solution.
        probability (float): The probability of accepting the solution.
    """
    if probability > 0.1:
        # Generate a new solution based on the current solution and the probability
        new_solution = solution + random.uniform(-1.0, 1.0)
        
        # Evaluate the new solution
        new_evaluations = 0
        while new_evaluations < 10:
            new_solution = np.random.uniform(-5.0, 5.0, new_solution.shape)
            new_evaluations += 1
            func(new_solution)
            
            # Accept the new solution with a probability less than 1
            if new_evaluations > 0 and new_evaluations < 10:
                new_probability = np.exp((new_evaluations - 1) / 10)
                if new_probability < probability:
                    new_solution = new_solution

        # Update the solution with the new solution
        solution = new_solution

        # Update the probability of acceptance based on the new solution
        new_probability = np.exp((10 - 1) / 10)
        if new_probability < probability:
            solution = new_solution

    return solution

optimal_solution = refine_solution(optimal_solution, 0.1)
print("Refined optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)