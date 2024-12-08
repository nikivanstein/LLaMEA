import numpy as np

class BlackBoxOptimizer:
    """
    An evolutionary algorithm for solving black box optimization problems.

    The algorithm uses a combination of evolutionary strategies, including mutation, selection, and crossover, to find the optimal solution.
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
                    # Update the solution with a mutation strategy
                    solution = np.random.uniform(-5.0, 5.0, self.dim)
                    # Add a crossover strategy to combine the mutated solution with another solution
                    if evaluations < self.budget - 1:
                        crossover_solution = np.random.uniform(-5.0, 5.0, self.dim)
                        solution = np.concatenate((crossover_solution, solution))

            # Update the solution with a selection strategy
            if evaluations < self.budget - 1:
                # Select the fittest solution
                fittest_solution = np.argmax(np.abs(func(solution)))
                solution = solution[fittest_solution]

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)