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
                    # Refine the solution strategy using a modified genetic algorithm
                    # Update the solution with the best solution found so far
                    # and add a constraint to avoid local optima
                    solution = np.random.uniform(-5.0, 5.0, self.dim)
                    if evaluations > 0:
                        # Calculate the fitness of the new solution
                        new_fitness = func(solution)
                        # Accept the new solution with a probability less than 1
                        if np.random.rand() < np.exp((new_fitness - func(solution)) / self.budget):
                            solution = solution

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Modified genetic algorithm to refine the solution strategy
def modified_genetic_algorithm(individual):
    """
    A modified genetic algorithm to refine the solution strategy.

    Args:
        individual (numpy array): The current individual.

    Returns:
        numpy array: The refined individual.
    """
    # Calculate the fitness of the individual
    fitness = func(individual)

    # Calculate the probability of accepting the individual
    probability = np.exp((fitness - func(np.random.uniform(-5.0, 5.0, individual.shape))) / 10)

    # Accept the individual with a probability less than 1
    if np.random.rand() < probability:
        return individual
    else:
        # Refine the individual strategy using a modified genetic algorithm
        # Update the individual with the best solution found so far
        # and add a constraint to avoid local optima
        individual = np.random.uniform(-5.0, 5.0, individual.shape)
        if num_evaluations > 0:
            # Calculate the fitness of the new individual
            new_fitness = func(individual)
            # Accept the new individual with a probability less than 1
            if np.random.rand() < np.exp((new_fitness - fitness) / 10):
                individual = individual


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimized_solution = BlackBoxOptimizer(100, 10).func(func)
print("Optimized solution:", optimized_solution)
print("Number of function evaluations:", BlackBoxOptimizer(100, 10).budget)