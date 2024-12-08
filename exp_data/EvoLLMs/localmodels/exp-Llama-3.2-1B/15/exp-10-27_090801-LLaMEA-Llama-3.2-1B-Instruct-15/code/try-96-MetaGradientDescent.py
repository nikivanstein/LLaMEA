import numpy as np
import random

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# Description: Meta-Heuristic Optimization using Evolutionary Strategies for Black Box Optimization
# Code: 
# ```python
def optimize_bbob(budget, dim, func, mutation_rate=0.01):
    """
    Meta-Heuristic Optimization using Evolutionary Strategies for Black Box Optimization.

    Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the problem.
        func (callable): The black box function to optimize.
        mutation_rate (float, optional): The mutation rate for the evolutionary strategy. Defaults to 0.01.

    Returns:
        tuple: A tuple containing the optimized parameter values and the objective function value.
    """
    # Initialize the population with random parameter values
    population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

    # Initialize the best solution and its fitness
    best_solution = None
    best_fitness = float('inf')

    while len(population) > 0 and budget > 0:
        # Select the fittest individual using tournament selection
        tournament_size = 10
        tournament_selection = random.choices(population, k=tournament_size)
        tournament_fitness = np.array([func(t) for t in tournament_selection])
        tournament_fitness /= tournament_size

        # Select the best individual using roulette wheel selection
        roulette_wheel = np.random.rand()
        cumulative_probability = 0.0
        for i, fitness in enumerate(tournament_fitness):
            cumulative_probability += fitness
            if roulette_wheel <= cumulative_probability:
                best_solution = tournament_selection[i]
                break

        # Evaluate the fitness of the best individual
        fitness = func(best_solution)

        # Update the population and the best solution
        population = [t + mutation_rate * (np.random.normal(0, 1, dim) - np.mean(best_solution)) for t in population]
        best_solution = best_solution if fitness < best_fitness else best_solution
        best_fitness = fitness

        # Update the budget
        budget -= 1

    # Return the best solution and its fitness
    return best_solution, best_fitness

# Test the optimization algorithm
func = lambda x: x**2
budget = 1000
dim = 10
optimized_solution, optimized_fitness = optimize_bbob(budget, dim, func)

# Print the results
print(f"Optimized solution: {optimized_solution}")
print(f"Optimized fitness: {optimized_fitness}")