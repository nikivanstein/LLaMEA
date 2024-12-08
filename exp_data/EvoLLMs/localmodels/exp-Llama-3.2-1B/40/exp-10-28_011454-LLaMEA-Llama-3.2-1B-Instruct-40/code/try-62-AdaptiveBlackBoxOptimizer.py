# Description: Black Box Optimization using Evolutionary Algorithm with Adaptive Strategy
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Adaptive Black Box Optimizer.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the optimization space.
        """
        self.budget = budget
        self.dim = dim
        self.population = None
        self.evaluations = None
        self.select_strategy = None

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
            self.select_strategy = self.select_fittest(pop_size, evaluations)

            # Mutate the selected individuals based on their mean and standard deviation
            self.population = self.mutate(self.population, evaluations)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < self.budget:
                break

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

    def select_fittest(self, pop_size, evaluations):
        """
        Select the fittest individuals in the population based on their mean and standard deviation.

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

    def mutate(self, population, evaluations):
        """
        Mutate the selected individuals based on their mean and standard deviation.

        Parameters:
        population (np.ndarray): The selected individuals.
        evaluations (list): The function values of the individuals in the population.

        Returns:
        np.ndarray: The mutated individuals.
        """
        # Create a copy of the population
        mutated = population.copy()

        # Randomly swap two individuals in the population based on their mean and standard deviation
        for i in range(len(mutated)):
            j = np.random.choice(len(mutated))
            if evaluations[i] > evaluations[j]:
                mutated[i], mutated[j] = mutated[j], mutated[i]

        return mutated

    def update_strategy(self, func, budget):
        """
        Update the strategy of the evolutionary algorithm based on the mean and standard deviation of the function values.

        Parameters:
        func (callable): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population and the number of evaluations
        self.population = np.random.uniform(-5.0, 5.0, (100, self.dim))
        self.evaluations = []

        # Run the evolutionary algorithm
        for gen in range(100):
            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(100)]

            # Select the fittest individuals based on their mean and standard deviation
            self.select_strategy = self.select_fittest(100, evaluations)

            # Mutate the selected individuals based on their mean and standard deviation
            self.population = self.mutate(self.population, evaluations)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(100)]

            # Check if the population has reached the budget
            if len(evaluations) < budget:
                break

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]