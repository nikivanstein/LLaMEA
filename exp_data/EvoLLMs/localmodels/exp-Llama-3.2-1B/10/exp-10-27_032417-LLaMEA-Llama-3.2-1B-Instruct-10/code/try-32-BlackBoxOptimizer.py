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

        # Initialize temperature and temperature schedule
        self.temperature = 1.0
        self.temperature_schedule = [self.temperature] * self.dim

        # Iterate over the range of possible solutions
        while evaluations < self.budget:
            # Generate a random solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the black box function at the current solution
            evaluations += 1
            func(solution)

            # Calculate the probability of accepting the current solution
            probabilities = np.exp((evaluations - evaluations) / self.budget)

            # Accept the current solution with a probability less than 1
            if random.random() < probabilities[0]:
                solution = solution

            # Update the temperature using the temperature schedule
            self.update_temperature()

            # If the current solution is better than the previous best solution, update the solution
            if evaluations > 0 and evaluations < self.budget:
                if evaluations > 0:
                    # Calculate the probability of accepting the current solution
                    probability = np.exp((evaluations - evaluations) / self.budget)

                    # Accept the current solution with a probability less than 1
                    if random.random() < probability:
                        solution = solution
                else:
                    # Update the solution with the best solution found so far
                    solution = None

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations

    def update_temperature(self):
        """
        Updates the temperature using the temperature schedule.
        """
        # Decrease the temperature by a factor
        self.temperature *= 0.9

        # Schedule the temperature decrease
        if self.temperature < 0.1:
            self.temperature_schedule = [self.temperature] * self.dim
        elif self.temperature > 0.1:
            self.temperature_schedule = [self.temperature] * self.dim
            self.temperature = 0.1