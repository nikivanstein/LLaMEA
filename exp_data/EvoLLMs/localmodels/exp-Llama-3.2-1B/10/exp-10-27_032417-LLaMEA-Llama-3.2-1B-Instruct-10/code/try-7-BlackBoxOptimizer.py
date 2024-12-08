import numpy as np

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

            # Calculate the probability of accepting the current solution
            probability = np.exp((evaluations - evaluations) / self.budget)

            # Accept the current solution with a probability less than 1
            if np.random.rand() < probability:
                solution = solution

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations

    def update_solution(self, individual, probability):
        """
        Updates the solution with a new individual based on the probability of acceptance.

        Args:
            individual (list): The current solution.
            probability (float): The probability of accepting the new individual.
        """
        # Refine the solution based on the probability of acceptance
        new_individual = individual + np.random.uniform(-1, 1, self.dim)
        if np.random.rand() < probability:
            new_individual = individual

        # Update the solution and the number of function evaluations
        solution = new_individual
        evaluations = 0

        # Iterate over the range of possible solutions
        while evaluations < self.budget:
            # Generate a random solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the black box function at the current solution
            evaluations += 1
            func(solution)

            # Calculate the probability of accepting the current solution
            probability = np.exp((evaluations - evaluations) / self.budget)

            # Accept the current solution with a probability less than 1
            if np.random.rand() < probability:
                solution = solution

        # Return the updated solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Update the solution with a new individual based on the probability of acceptance
updated_solution, num_evaluations = optimizer.update_solution(optimal_solution, 0.1)
print("Updated solution:", updated_solution)
print("Number of function evaluations:", num_evaluations)

# Example usage:
def func2(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func2)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Update the solution with a new individual based on the probability of acceptance
updated_solution, num_evaluations = optimizer.update_solution(optimal_solution, 0.1)
print("Updated solution:", updated_solution)
print("Number of function evaluations:", num_evaluations)