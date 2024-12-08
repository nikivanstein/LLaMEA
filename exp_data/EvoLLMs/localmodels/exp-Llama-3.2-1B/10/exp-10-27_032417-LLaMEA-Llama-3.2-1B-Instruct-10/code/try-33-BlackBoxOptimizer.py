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
        self.population = self.initialize_population()
        self.temperature = 1000

    def initialize_population(self):
        """
        Initializes the population with random solutions within the search space.

        Returns:
            list: A list of solutions.
        """
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

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
            solution = self.population[np.random.choice(self.population_size)]

            # Evaluate the black box function at the current solution
            evaluations += 1
            func(solution)

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = 1 - np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if np.random.rand() < probability:
                        solution = solution
                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations

    def update_solution(self, solution):
        """
        Updates the solution with a new random solution within the search space.

        Args:
            solution (list): The new solution.
        """
        # Initialize the new solution
        new_solution = None

        # Generate a new solution within the search space
        for _ in range(self.population_size):
            new_solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Check if the new solution is better than the current solution
            if new_solution!= solution:
                # Calculate the probability of accepting the new solution
                probability = 1 - np.exp((self.budget - evaluations) / self.budget)

                # Accept the new solution with a probability less than 1
                if np.random.rand() < probability:
                    new_solution = new_solution

        # Update the solution
        self.population[np.random.choice(self.population_size)] = new_solution

    def __str__(self):
        """
        Returns a string representation of the optimizer.

        Returns:
            str: A string representation of the optimizer.
        """
        return f"BlackBoxOptimizer(budget={self.budget}, dim={self.dim}, population_size={self.population_size})"


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Update the solution with a new random solution
optimizer.update_solution(optimal_solution)
print("Updated solution:", optimizer.population[np.random.choice(optimizer.population_size)])
print("Updated number of function evaluations:", optimizer.budget)