# Description: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
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

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        # Refine the solution with a new individual
                        new_individual = np.random.uniform(-5.0, 5.0, self.dim)
                        new_individual = new_individual + 0.1 * (solution - new_individual)
                        solution = new_individual
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

# Novel heuristic algorithm: Genetic Algorithm for Black Box Optimization with Simulated Annealing
# Description: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
import numpy as np
import random

class GeneticBlackBoxOptimizer:
    """
    A novel metaheuristic algorithm for solving black box optimization problems.

    The algorithm combines genetic algorithm and simulated annealing to find the optimal solution.
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
        # Initialize the population and the number of function evaluations
        population = []
        for _ in range(100):
            # Generate an initial population of random solutions
            population.append(np.random.uniform(-5.0, 5.0, self.dim))

            # Evaluate the black box function at each solution in the population
            for individual in population:
                func(individual)

            # Select the fittest solutions to reproduce
            fittest_individuals = sorted(population, key=lambda x: func(x), reverse=True)[:self.budget // 2]

            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(self.budget - len(fittest_individuals)):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = (parent1 + parent2) / 2
                if random.random() < 0.1:
                    child = random.uniform(-5.0, 5.0)
                new_population.append(child)

            # Replace the least fit individuals in the population with the new population
            population = new_population

        # Return the optimal solution and the number of function evaluations used
        return population[0], len(population)


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = GeneticBlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)