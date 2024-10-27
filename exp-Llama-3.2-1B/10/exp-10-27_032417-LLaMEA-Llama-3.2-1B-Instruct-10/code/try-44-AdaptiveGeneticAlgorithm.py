import numpy as np
import random

class AdaptiveGeneticAlgorithm:
    """
    An adaptive genetic algorithm for solving black box optimization problems.

    The algorithm uses a combination of genetic algorithm and simulated annealing to find the optimal solution.
    """

    def __init__(self, budget, dim):
        """
        Initializes the adaptive genetic algorithm with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimizes a black box function using the adaptive genetic algorithm.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the solution and the number of function evaluations
        solution = None
        evaluations = 0

        # Initialize the current best solution and its fitness
        best_solution = None
        best_fitness = float('-inf')

        # Iterate over the range of possible solutions
        while evaluations < self.budget:
            # Generate a random solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the black box function at the current solution
            evaluations += 1
            func(solution)

            # If the current solution is better than the previous best solution, update the best solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = solution

                    # If the current solution is better than the best solution found so far, update the best solution
                    if evaluations > 0 and evaluations < self.budget:
                        if evaluations > 0:
                            # Calculate the probability of accepting the current solution
                            probability = np.exp((evaluations - evaluations) / self.budget)

                            # Accept the current solution with a probability less than 1
                            if np.random.rand() < probability:
                                best_solution = solution
                                best_fitness = func(best_solution)
                        else:
                            # Update the best solution with the best solution found so far
                            best_solution = None
                            best_fitness = float('-inf')

        # Return the optimal solution and the number of function evaluations used
        return best_solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = AdaptiveGeneticAlgorithm(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Adaptive genetic algorithm with adaptive probability of accepting the current solution
def func_adaptive(x):
    return x**2 + 2*x + 1

optimizer_adaptive = AdaptiveGeneticAlgorithm(100, 10)
optimal_solution_adaptive, num_evaluations_adaptive = optimizer_adaptive(func_adaptive)
print("Optimal solution (adaptive):", optimal_solution_adaptive)
print("Number of function evaluations (adaptive):", num_evaluations_adaptive)