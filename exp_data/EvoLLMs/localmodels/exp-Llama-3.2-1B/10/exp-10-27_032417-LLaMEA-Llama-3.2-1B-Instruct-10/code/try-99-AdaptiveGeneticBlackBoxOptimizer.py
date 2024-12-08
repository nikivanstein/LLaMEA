import numpy as np

class AdaptiveGeneticBlackBoxOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems with adaptive probability of acceptance.
    """

    def __init__(self, budget, dim, adaptive_prob=0.1):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            adaptive_prob (float, optional): The probability of accepting the current solution. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.adaptive_prob = adaptive_prob

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
            prob = np.random.rand() / self.adaptive_prob

            # Accept the current solution with a probability less than 1
            if np.random.rand() < prob:
                solution = solution
            else:
                # Update the solution with the best solution found so far
                solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = AdaptiveGeneticBlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)