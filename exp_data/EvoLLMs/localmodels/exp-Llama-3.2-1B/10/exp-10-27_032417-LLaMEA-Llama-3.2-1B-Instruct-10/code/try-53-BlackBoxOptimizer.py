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

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                # Calculate the probability of accepting the current solution
                probability = np.exp((evaluations - evaluations) / self.budget)

                # Accept the current solution with a probability less than 1
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

def refine_solution(individual, budget):
    """
    Refines the selected solution by changing its strategy.

    Args:
        individual (list): The selected solution.
        budget (int): The remaining budget for the solution refinement.

    Returns:
        tuple: A tuple containing the refined solution and the number of function evaluations used.
    """
    # Define the probability of changing the strategy
    probability = 0.1

    # Generate a new solution by changing the strategy
    new_individual = individual.copy()
    for i in range(len(individual)):
        # Change the strategy for each dimension
        if np.random.rand() < probability:
            new_individual[i] += np.random.uniform(-5.0, 5.0)

    # Evaluate the black box function at the new solution
    func(new_individual)

    # If the new solution is better than the previous best solution, update the solution
    if new_individual[0] > individual[0]:
        individual = new_individual

    # Return the refined solution and the number of function evaluations used
    return individual, budget - 1


optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
refined_solution, num_evaluations = refine_solution(optimal_solution, num_evaluations)

print("Optimal solution:", optimal_solution)
print("Refined solution:", refined_solution)
print("Number of function evaluations:", num_evaluations)