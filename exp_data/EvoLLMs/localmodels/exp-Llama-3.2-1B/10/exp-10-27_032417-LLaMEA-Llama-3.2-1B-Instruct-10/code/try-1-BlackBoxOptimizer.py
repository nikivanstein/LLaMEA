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
                # Calculate the probability of accepting the current solution
                probability = np.exp((evaluations - evaluations) / self.budget)

                # Accept the current solution with a probability less than 1
                if np.random.rand() < probability:
                    solution = solution
                else:
                    # Refine the solution using a modified genetic algorithm
                    if evaluations < self.budget // 2:
                        # Use a simple genetic algorithm to refine the solution
                        def genetic_algorithm(individual):
                            fitness = func(individual)
                            best_solution = None
                            best_fitness = float('-inf')
                            for i in range(self.dim):
                                new_individual = individual.copy()
                                new_individual[i] += random.uniform(-1, 1)
                                fitness_value = func(new_individual)
                                if fitness_value > best_fitness:
                                    best_solution = new_individual
                                    best_fitness = fitness_value
                            return best_solution
                        solution = genetic_algorithm(solution)
                    else:
                        # Use simulated annealing to refine the solution
                        def simulated_annealing(individual):
                            fitness = func(individual)
                            temperature = 1.0
                            for _ in range(self.dim):
                                new_individual = individual.copy()
                                new_individual[i] += random.uniform(-1, 1)
                                fitness_value = func(new_individual)
                                if fitness_value > best_fitness:
                                    best_solution = new_individual
                                    best_fitness = fitness_value
                                elif np.random.rand() < np.exp((best_fitness - fitness_value) / temperature):
                                    best_solution = new_individual
                                    temperature *= 0.99
                            return best_solution
                        solution = simulated_annealing(solution)

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)