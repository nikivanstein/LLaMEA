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

    def simulated_annealing(self, func, initial_solution, temperature, cooling_rate):
        """
        Simulated Annealing Algorithm for Optimization.

        Args:
            func (function): The black box function to optimize.
            initial_solution (numpy array): The initial solution.
            temperature (float): The initial temperature.
            cooling_rate (float): The cooling rate.

        Returns:
            tuple: A tuple containing the optimal solution and the number of function evaluations used.
        """
        # Initialize the solution and the number of function evaluations
        solution = initial_solution
        evaluations = 0

        # Initialize the current solution
        current_solution = solution

        # Iterate until the temperature reaches zero
        while temperature > 0.1:
            # Generate a new solution within the search space
            new_solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the black box function at the new solution
            evaluations += 1
            func(new_solution)

            # Calculate the probability of accepting the new solution
            probability = np.exp((evaluations - evaluations) / temperature)

            # Accept the new solution with a probability less than 1
            if np.random.rand() < probability:
                current_solution = new_solution

            # Update the solution with the best solution found so far
            if evaluations > 0:
                # Calculate the probability of accepting the current solution
                probability = np.exp((evaluations - evaluations) / temperature)

                # Accept the current solution with a probability less than 1
                if np.random.rand() < probability:
                    current_solution = solution

            # Decrease the temperature
            temperature *= cooling_rate

        # Return the optimal solution and the number of function evaluations used
        return current_solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Example usage of Simulated Annealing Algorithm
initial_solution = np.array([-5.0, -5.0])
optimal_solution, num_evaluations = optimizer.simulated_annealing(func, initial_solution, 1000, 0.99)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)