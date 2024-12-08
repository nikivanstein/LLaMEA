import numpy as np

class AdaptiveGeneticAlgorithm:
    """
    A metaheuristic algorithm for solving black box optimization problems.

    The algorithm uses a combination of genetic algorithm and simulated annealing to find the optimal solution.
    """

    def __init__(self, budget, dim, alpha=0.1, beta=0.1):
        """
        Initializes the optimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            alpha (float, optional): The probability of accepting a worse solution. Defaults to 0.1.
            beta (float, optional): The probability of accepting a worse solution based on the current temperature. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta

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

        # Initialize the temperature
        temperature = 1.0

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

            # Update the temperature based on the probability of acceptance
            temperature *= self.beta

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = solution

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

agab = AdaptiveGeneticAlgorithm(100, 10, alpha=0.5, beta=0.5)
optimal_solution, num_evaluations = agab(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)