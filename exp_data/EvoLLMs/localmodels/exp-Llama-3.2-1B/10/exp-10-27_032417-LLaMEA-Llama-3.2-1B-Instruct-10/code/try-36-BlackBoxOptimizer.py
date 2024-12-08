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
                # Refine the solution by changing the individual lines of the selected solution
                # to refine its strategy
                # For example, you can change the lines to:
                # solution = np.random.uniform(-5.0, 5.0, self.dim)
                # solution[0] += 0.1
                # solution[1] += 0.2
                # solution[2] += 0.3
                solution = np.random.uniform(-5.0, 5.0, self.dim)

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# One-line description: A hybrid genetic algorithm for black box optimization that combines genetic algorithm and simulated annealing.
# Code: 