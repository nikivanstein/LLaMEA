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

            # Select the fittest individuals
            self.select_strategy = self.select_fittest(pop_size, evaluations)

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

            # Select the fittest individuals
            self.select_strategy = self.select_fittest(100, evaluations)

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(100)]

            # Check if the population has reached the budget
            if len(evaluations) < budget:
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
        if np.random.rand() < 0.4:
            # Use the current strategy
            return self.select_strategy, func
        else:
            # Use the adaptive strategy
            return self.adaptive_strategy, func

# Description: Adaptive Black Box Optimization using Evolutionary Algorithm with Adaptive Strategy
# Code: 
# ```python
def adaptive_black_box_optimize(func, budget, dim, population_size=100, num_generations=100):
    """
    Optimize the black box function using evolutionary algorithm with adaptive strategy.

    Parameters:
    func (callable): The black box function to optimize.
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the optimization space.
    population_size (int): The size of the population. Defaults to 100.
    num_generations (int): The number of generations. Defaults to 100.

    Returns:
    tuple: The optimized parameters and the optimized function value.
    """
    # Initialize the AdaptiveBlackBoxOptimizer
    optimizer = AdaptiveBlackBoxOptimizer(budget, dim)

    # Run the evolutionary algorithm
    for gen in range(num_generations):
        # Evaluate the function at each individual in the population
        evaluations = [func(optimizer.population[i]) for i in range(optimizer.population_size)]

        # Select the fittest individuals
        optimizer.select_strategy = optimizer.select_fittest(optimizer.population_size, evaluations)

        # Mutate the selected individuals
        optimizer.population = optimizer.mutate(optimizer.population)

        # Evaluate the function at each individual in the population
        evaluations = [func(optimizer.population[i]) for i in range(optimizer.population_size)]

        # Check if the population has reached the budget
        if len(evaluations) < optimizer.budget:
            break

    # Update the adaptive strategy
    optimizer.adaptive_strategy = optimizer.update_adaptive_strategy(func, optimizer.budget)

    # Return the optimized parameters and the optimized function value
    return optimizer.population, optimizer.evaluations[-1]

# Example usage:
# Define a black box function
def func(x):
    return x**2 + 2*x + 1

# Optimize the function using evolutionary algorithm with adaptive strategy
optimized_params, optimized_func_value = adaptive_black_box_optimize(func, 1000, 2)

# Print the optimized parameters and function value
print("Optimized Parameters:", optimized_params)
print("Optimized Function Value:", optimized_func_value)