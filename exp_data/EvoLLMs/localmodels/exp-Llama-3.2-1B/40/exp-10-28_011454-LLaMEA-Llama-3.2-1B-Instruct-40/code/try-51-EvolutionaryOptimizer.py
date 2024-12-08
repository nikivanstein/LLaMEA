# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Evolutionary Optimizer.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the optimization space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function using evolutionary algorithm.

        Parameters:
        func (callable): The black box function to optimize.

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
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Select the fittest individuals based on their mean and standard deviation
            mean = np.mean(evaluations)
            std = np.std(evaluations)
            indices = np.argsort([mean - std * i for i in range(pop_size)])

            # Refine the selected individuals based on their mean and standard deviation
            refined_indices = np.random.choice(indices, pop_size, replace=True, p=[mean - std * i / pop_size for i in range(pop_size)])
            self.population = self.population[refined_indices]

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

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

    def mutate(self, population):
        """
        Mutate the selected individuals.

        Parameters:
        population (np.ndarray): The selected individuals.

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

def sphere_problem():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)

    # Initialize the evolutionary optimizer
    optimizer = EvolutionaryOptimizer(100, 5)

    # Optimize the function
    optimized_params, optimized_function_value = optimizer(func)

    # Return the optimized parameters and the optimized function value
    return optimized_params, optimized_function_value

# Test the function
optimized_params, optimized_function_value = sphere_problem()
print(f"Optimized Parameters: {optimized_params}")
print(f"Optimized Function Value: {optimized_function_value}")