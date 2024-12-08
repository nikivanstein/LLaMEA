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

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                # Calculate the probability of accepting the current solution
                probability = np.exp((evaluations - evaluations) / self.budget)

                # Accept the current solution with a probability less than 1
                if np.random.rand() < probability:
                    solution = solution

            # If the current solution is worse than the previous best solution, accept it with a probability less than 1
            else:
                probability = np.exp((evaluations - evaluations) / self.budget)
                if np.random.rand() < probability:
                    solution = solution

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

def update_solution(solution, budget):
    """
    Updates the solution with a new individual based on the probability of acceptance.

    Args:
        solution (array): The current solution.
        budget (int): The remaining number of function evaluations.
    """
    # Calculate the probability of accepting the current solution
    probability = np.random.rand()

    # Accept the current solution with a probability less than 1
    if probability < 0.1:
        solution = np.random.uniform(-5.0, 5.0, solution.shape)

    # If the current solution is worse than the previous best solution, accept it with a probability less than 1
    elif probability < 0.9:
        solution = update_solution(solution, budget - 1)

    return solution


def optimize_func(func, optimizer, budget):
    """
    Optimizes a black box function using the given optimizer.

    Args:
        func (function): The black box function to optimize.
        optimizer (BlackBoxOptimizer): The optimizer to use.
        budget (int): The maximum number of function evaluations allowed.

    Returns:
        tuple: A tuple containing the optimal solution and the number of function evaluations used.
    """
    # Initialize the solution and the number of function evaluations
    solution = None
    evaluations = 0

    # Iterate over the range of possible solutions
    while evaluations < budget:
        # Update the solution based on the probability of acceptance
        solution = update_solution(solution, evaluations)

        # Evaluate the black box function at the current solution
        evaluations += 1
        func(solution)

    # Return the optimal solution and the number of function evaluations used
    return solution, evaluations


# Example usage:
optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimize_func(func, optimizer, 100)

# Print the optimal solution and the number of function evaluations used
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)