# Description: Adaptive Black Box Optimization using Evolutionary Algorithm with Adaptive Strategy
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
        self.adaptive_strategy = None

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

            # Select the fittest individuals using the adaptive strategy
            self.select_strategy = self.select_adaptive_strategy(func, self.budget, self.population)

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < self.budget:
                break

        # Update the adaptive strategy
        self.adaptive_strategy = self.select_adaptive_strategy(func, self.budget, self.population)

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

    def select_adaptive_strategy(self, func, budget, population):
        """
        Select the adaptive strategy of the evolutionary algorithm.

        Parameters:
        func (callable): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        population (np.ndarray): The population of individuals.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Define the adaptive strategy
        if np.random.rand() < 0.6:
            # Use the current strategy
            return self.select_strategy, func
        else:
            # Use the adaptive strategy
            return self.adaptive_strategy, func

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

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolutionary Algorithm with Adaptive Strategy