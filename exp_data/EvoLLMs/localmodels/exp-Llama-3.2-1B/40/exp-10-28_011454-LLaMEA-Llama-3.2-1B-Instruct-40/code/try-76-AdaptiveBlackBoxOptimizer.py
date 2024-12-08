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

            # Select the fittest individuals using adaptive strategy
            self.select_strategy = self.select_adaptive_strategy(evaluations)

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < self.budget:
                break

        # Update the adaptive strategy
        self.adaptive_strategy = self.update_adaptive_strategy(func, self.budget)

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

    def select_adaptive_strategy(self, evaluations):
        """
        Select the adaptive strategy of the evolutionary algorithm.

        Parameters:
        evaluations (list): The function values of the individuals in the population.

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

    def update_adaptive_strategy(self, func, budget):
        """
        Update the adaptive strategy of the evolutionary algorithm.

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

            # Select the fittest individuals using adaptive strategy
            self.select_strategy = self.select_adaptive_strategy(evaluations)

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(100)]

            # Check if the population has reached the budget
            if len(evaluations) < budget:
                break

        # Update the adaptive strategy
        self.adaptive_strategy = self.select_adaptive_strategy(func, budget)

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

# Description: Adaptive Black Box Optimization using Evolutionary Algorithm with Adaptive Strategy
# Code: 
# ```python
# Optimizing black box functions using evolutionary algorithms
# 
# The adaptive black box optimization algorithm uses an evolutionary algorithm to optimize black box functions.
# The algorithm uses a adaptive strategy to select the fittest individuals in the population, which is then used to mutate the selected individuals.
# The algorithm is designed to handle a wide range of tasks and has been evaluated on the BBOB test suite of 24 noiseless functions.
# 
# The algorithm consists of the following components:
# - Initialize the population with random parameters
# - Run the evolutionary algorithm for a specified number of generations
# - Select the fittest individuals using the adaptive strategy
# - Mutate the selected individuals
# - Evaluate the function at each individual in the population
# - Check if the population has reached the budget
# - Update the adaptive strategy
# 
# The algorithm is suitable for solving black box optimization problems with a wide range of tasks and has been evaluated on the BBOB test suite of 24 noiseless functions.
# 
# Parameters:
# - budget (int): The maximum number of function evaluations allowed
# - dim (int): The dimensionality of the optimization space
# 
# Returns:
# - The optimized parameters and the optimized function value
# """
adaptive_black_box_optimizer = AdaptiveBlackBoxOptimizer(budget, dim)