import numpy as np
import random
import copy

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
                        # Refine the solution using a mutation strategy
                        mutated_solution = solution.copy()
                        mutated_solution[random.randint(0, self.dim - 1)] = random.uniform(-5.0, 5.0)
                        solution = mutated_solution
                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Mutation strategy to refine the solution
def mutate(solution, dim):
    """
    Mutates a solution using a random permutation.

    Args:
        solution (numpy array): The current solution.
        dim (int): The dimensionality of the search space.

    Returns:
        numpy array: The mutated solution.
    """
    return solution[np.random.randint(0, dim)]


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Refine the solution using the mutation strategy
mutated_solution = mutate(optimal_solution, 10)
print("Refined solution:", mutated_solution)

# Simulate the evolutionary algorithm
def evolutionary_algorithm(func, optimizer, mutation_rate):
    """
    Simulates the evolutionary algorithm.

    Args:
        func (function): The black box function to optimize.
        optimizer (BlackBoxOptimizer): The optimizer to use.
        mutation_rate (float): The probability of mutation.

    Returns:
        tuple: A tuple containing the optimal solution and the number of function evaluations used.
    """
    # Initialize the population with random solutions
    population = [copy.deepcopy(optimal_solution) for _ in range(100)]

    # Evaluate the fitness of each solution
    for _ in range(100):
        fitness = [func(solution) for solution in population]
        population = [solution for _, solution in sorted(zip(fitness, population), reverse=True)]

    # Select the fittest solutions
    fittest_solutions = [solution for _, solution in sorted(zip(fitness, population), reverse=True)]

    # Mutate the fittest solutions
    for solution in fittest_solutions:
        mutated_solution = mutate(solution, 10)
        population.append(mutated_solution)

    # Evolve the population
    for _ in range(10):
        # Evaluate the fitness of each solution
        fitness = [func(solution) for solution in population]
        population = [solution for _, solution in sorted(zip(fitness, population), reverse=True)]

    # Select the fittest solutions
    fittest_solutions = [solution for _, solution in sorted(zip(fitness, population), reverse=True)]

    # Return the fittest solution
    return fittest_solutions[0]


# Example usage:
fittest_solution = evolutionary_algorithm(func, optimizer, 0.1)
print("Fittest solution:", fittest_solution)
print("Number of function evaluations:", len(fittest_solution))