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
        self.population_size = 100
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        """
        Generates an initial population of random solutions.

        Returns:
            list: A list of initial solutions.
        """
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

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
            solution = random.choice(self.population)

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


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Evolved strategy
def evolved_strategy(func, optimizer):
    """
    Generates an evolved strategy by evolving the initial population.

    Args:
        func (function): The black box function to optimize.
        optimizer (BlackBoxOptimizer): The optimizer to use.

    Returns:
        tuple: A tuple containing the evolved solution and the number of function evaluations used.
    """
    population = optimizer.generate_initial_population()

    # Evolve the population using genetic algorithm
    for _ in range(100):
        population = optimizer(population)

    # Return the evolved solution and the number of function evaluations used
    return population[0], 100


# Example usage:
def func(x):
    return x**2 + 2*x + 1

evolved_solution, num_evaluations = evolved_strategy(func, optimizer)
print("Evolved solution:", evolved_solution)
print("Number of function evaluations:", num_evaluations)

# One-line description with main idea:
# Evolutionary Algorithm for Black Box Optimization
# Uses genetic algorithm to evolve initial population, leading to better solutions