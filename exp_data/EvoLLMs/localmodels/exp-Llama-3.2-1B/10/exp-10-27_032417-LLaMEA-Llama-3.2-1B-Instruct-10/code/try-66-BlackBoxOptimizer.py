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

                    # If the current solution is better than the previous best solution, update it
                    if evaluations > 0 and evaluations < self.budget:
                        # Calculate the probability of accepting the current solution
                        probability = np.exp((evaluations - evaluations) / self.budget)

                        # Accept the current solution with a probability less than 1
                        if np.random.rand() < probability:
                            solution = np.random.uniform(-5.0, 5.0, self.dim)

            # Refine the solution based on the current best solution
            if solution is not None and evaluations > 0 and evaluations < self.budget:
                # Calculate the distance between the current solution and the previous best solution
                distance = np.linalg.norm(solution - self.best_solution)

                # Refine the solution based on the distance to the previous best solution
                if distance < 0.1:
                    solution = solution

        # Return the optimal solution and the number of function evaluations used
        return solution, evaluations


# Example usage:
def func(x):
    return x**2 + 2*x + 1

def optimize_func(x):
    return x

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution, num_evaluations = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Number of function evaluations:", num_evaluations)

# Adapt the solution based on the current best solution
def adapt_solution(solution, best_solution):
    if np.linalg.norm(solution - best_solution) < 0.1:
        return solution
    else:
        return best_solution

optimal_solution = adapt_solution(optimal_solution, optimize_func(optimal_solution))
print("Optimal solution after adaptation:", optimal_solution)

# Refine the solution based on the current best solution
def refine_solution(solution, best_solution):
    if np.linalg.norm(solution - best_solution) < 0.1:
        return np.random.uniform(-5.0, 5.0, self.dim)
    else:
        return best_solution

best_solution = optimize_func(optimal_solution)
optimal_solution = refine_solution(optimal_solution, best_solution)
print("Optimal solution after refinement:", optimal_solution)

# Example usage with different dimensionality
def func_dim_10(x):
    return x**2 + 2*x + 1

def optimize_func_dim_10(x):
    return x

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution_dim_10, num_evaluations_dim_10 = optimizer(func_dim_10)
print("Optimal solution for dimension 10:", optimal_solution_dim_10)
print("Number of function evaluations for dimension 10:", num_evaluations_dim_10)

# Example usage with different number of function evaluations
def func_eval_100(x):
    return x**2 + 2*x + 1

def optimize_func_eval_100(x):
    return x

optimizer = BlackBoxOptimizer(100, 10)
optimal_solution_eval_100, num_evaluations_eval_100 = optimizer(func_eval_100)
print("Optimal solution for function evaluations 100:", optimal_solution_eval_100)
print("Number of function evaluations for function evaluations 100:", num_evaluations_eval_100)

# Example usage with different number of noiseless functions
def func_eval_24(x):
    return x**2 + 2*x + 1

def optimize_func_eval_24(x):
    return x

optimizer = BlackBoxOptimizer(100, 24)
optimal_solution_eval_24, num_evaluations_eval_24 = optimizer(func_eval_24)
print("Optimal solution for function evaluations 24:", optimal_solution_eval_24)
print("Number of function evaluations for function evaluations 24:", num_evaluations_eval_24)