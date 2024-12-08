# Description: Refines the strategy of the Black Box Optimization using Evolutionary Algorithm
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

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

    def __call__(self, func, **kwargs):
        """
        Optimize the black box function using evolutionary algorithm.

        Parameters:
        func (callable): The black box function to optimize.
        **kwargs: Additional arguments for the optimization function.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size and the number of generations
        pop_size = 100
        num_generations = 100

        # Initialize the population with random parameters
        self.population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))

        # Run the evolutionary algorithm
        for gen in range(num_generations):
            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i], **kwargs) for i in range(pop_size)]

            # Select the fittest individuals
            self.population = self.select_fittest(pop_size, evaluations, **kwargs)

            # Mutate the selected individuals
            self.population = self.mutate(self.population, **kwargs)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i], **kwargs) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < self.budget:
                break

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

    def select_fittest(self, pop_size, evaluations, **kwargs):
        """
        Select the fittest individuals in the population.

        Parameters:
        pop_size (int): The size of the population.
        evaluations (list): The function values of the individuals in the population.
        **kwargs: Additional arguments for the selection function.

        Returns:
        np.ndarray: The indices of the fittest individuals.
        """
        # Calculate the mean and standard deviation of the function values
        mean = np.mean(evaluations)
        std = np.std(evaluations)

        # Select the fittest individuals based on their mean and standard deviation
        indices = np.argsort([mean - std * i for i in range(pop_size)])

        return indices

    def mutate(self, population, **kwargs):
        """
        Mutate the selected individuals.

        Parameters:
        population (np.ndarray): The selected individuals.
        **kwargs: Additional arguments for the mutation function.

        Returns:
        np.ndarray: The mutated individuals.
        """
        # Create a copy of the population
        mutated = population.copy()

        # Randomly swap two individuals in the population
        for i in range(len(mutated)):
            j = np.random.choice(len(mutated))
            mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated

# Example usage
if __name__ == "__main__":
    # Define the function to optimize
    def func(x):
        return x[0]**2 + 2*x[1]**2

    # Create an instance of the Black Box Optimizer
    optimizer = BlackBoxOptimizer(budget=100, dim=2)

    # Optimize the function using the evolutionary algorithm
    optimized_params, optimized_func_value = optimizer(func, **{"x": [0, 1]})

    # Print the optimized parameters and function value
    print("Optimized Parameters:", optimized_params)
    print("Optimized Function Value:", optimized_func_value)