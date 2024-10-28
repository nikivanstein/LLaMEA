import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation_cooling(self, func, initial_individual, budget, cooling_rate):
        """
        Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm

        Args:
            func (function): The black box function to optimize.
            initial_individual (list): The initial individual to start the algorithm with.
            budget (int): The maximum number of function evaluations allowed.
            cooling_rate (float): The rate at which the cooling factor decreases.

        Returns:
            list: The best individual found after the optimization process.
        """
        # Initialize the population with the initial individual
        population = [initial_individual]
        # Initialize the current best individual
        best_individual = initial_individual
        # Initialize the current best fitness value
        best_fitness = self.evaluate_fitness(func, initial_individual)

        # Repeat the optimization process for the specified number of iterations
        for _ in range(budget):
            # Generate a new individual by iterated permutation
            new_individual = self.iterated_permutation(func, initial_individual, self.dim)
            # Evaluate the new individual
            value = self.evaluate_fitness(func, new_individual)
            # Check if the new individual is better than the current best individual
            if value > best_fitness:
                # Update the current best individual and fitness value
                best_individual = new_individual
                best_fitness = value
            # Update the population with the new individual
            population.append(new_individual)
        # Return the best individual found
        return best_individual

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
def evaluateBBOB(func, individual, logger):
    # Evaluate the function at the individual
    value = func(individual)
    # Log the result
    logger.info(f"Value at {individual}: {value}")

# Initialize the optimizer with a budget of 1000 evaluations
optimizer = BlackBoxOptimizer(1000, 10)

# Optimize the BBOB function using the iterated permutation and cooling algorithm
best_individual = optimizer.iterated_permutation_cooling(evaluateBBOB, [-5.0, 5.0], 1000, 0.95)

# Print the best individual found
print(best_individual)