# Black Box Optimization using Evolutionary Algorithm with Refinement
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Black Box Optimizer.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the optimization space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func, initial_solution=None, iterations=100, mutation_rate=0.1):
        """
        Optimize the black box function using evolutionary algorithm.

        Parameters:
        func (callable): The black box function to optimize.
        initial_solution (list, optional): The initial solution. Defaults to None.
        iterations (int, optional): The number of iterations. Defaults to 100.
        mutation_rate (float, optional): The mutation rate. Defaults to 0.1.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size and the number of generations
        pop_size = 100
        num_generations = iterations

        # Initialize the population with random parameters
        self.population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

        # Run the evolutionary algorithm
        for gen in range(num_generations):
            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Select the fittest individuals
            self.population = self.select_fittest(pop_size, evaluations)

            # Mutate the selected individuals
            self.population = self.mutate(self.population, mutation_rate)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < self.budget:
                break

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

    def select_fittest(self, pop_size, evaluations):
        """
        Select the fittest individuals in the population.

        Parameters:
        pop_size (int): The size of the population.
        evaluations (list): The function values of the individuals in the population.

        Returns:
        np.ndarray: The indices of the fittest individuals.
        """
        # Calculate the mean and standard deviation of the function values
        mean = np.mean(evaluations)
        std = np.std(evaluations)

        # Select the fittest individuals based on their mean and standard deviation
        indices = np.argsort([mean - std * i for i in range(pop_size)])

        return indices

    def mutate(self, population, mutation_rate):
        """
        Mutate the selected individuals.

        Parameters:
        population (np.ndarray): The selected individuals.
        mutation_rate (float): The mutation rate.

        Returns:
        np.ndarray: The mutated individuals.
        """
        # Create a copy of the population
        mutated = population.copy()

        # Randomly swap two individuals in the population
        for i in range(len(mutated)):
            j = np.random.choice(len(mutated))
            mutated[i], mutated[j] = mutated[j], mutated[i]

        # Apply the mutation rate
        mutated = np.clip(mutated, -5.0, 5.0)

        return mutated

# Example usage:
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the Black Box Optimizer
    optimizer = BlackBoxOptimizer(budget=100, dim=5)

    # Optimize the function
    optimized_solution, optimized_function_value = optimizer(func, initial_solution=[1.0, 1.0, 1.0, 1.0, 1.0], iterations=1000)

    # Print the result
    print("Optimized solution:", optimized_solution)
    print("Optimized function value:", optimized_function_value)