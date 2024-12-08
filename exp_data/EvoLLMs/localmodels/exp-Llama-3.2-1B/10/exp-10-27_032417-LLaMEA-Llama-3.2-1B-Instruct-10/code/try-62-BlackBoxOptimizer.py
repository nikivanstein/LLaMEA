import numpy as np

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
            probability = np.exp((evaluations - evaluations) / self.budget)

            # Accept the current solution with a probability less than 1
            if np.random.rand() < probability:
                solution = solution

            # Refine the solution strategy based on the current probability
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = np.random.uniform(-5.0, 5.0, self.dim)
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

# Refine the solution strategy based on the probability of acceptance
def refine_solution(solution, probability):
    if probability < 0.1:
        # Increase the step size of the solution
        solution += np.random.uniform(-1.0, 1.0, self.dim)
    else:
        # Decrease the step size of the solution
        solution -= np.random.uniform(1.0, -1.0, self.dim)

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
refined_solution, num_evaluations = optimizer(func)
print("Refined optimal solution:", refined_solution)
print("Number of function evaluations:", num_evaluations)