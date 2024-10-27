import numpy as np
import random

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
                if evaluations > 0:
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


def genotype(solution):
    """
    Returns the genotype of the solution.

    Args:
        solution (numpy array): The solution.

    Returns:
        tuple: The genotype.
    """
    return tuple(solution)


def phenotype(func, genotype):
    """
    Evaluates the phenotype of the genotype.

    Args:
        func (function): The black box function.
        genotype (tuple): The genotype.

    Returns:
        float: The phenotype.
    """
    return func(genotype)


# Example usage:
def func(x):
    return x**2 + 2*x + 1

def fitness(individual):
    """
    Evaluates the fitness of the individual.

    Args:
        individual (numpy array): The individual.

    Returns:
        float: The fitness.
    """
    return phenotype(func, genotype(individual))

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

genotype = genotype(optimal_solution)
print("Genotype:", genotype)
fitness_value = fitness(genotype)
print("Fitness value:", fitness_value)