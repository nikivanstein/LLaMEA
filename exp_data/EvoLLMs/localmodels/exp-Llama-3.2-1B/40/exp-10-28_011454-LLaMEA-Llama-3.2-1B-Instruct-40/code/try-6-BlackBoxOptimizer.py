# Black Box Optimization using Evolutionary Algorithm with Refining Strategy
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

    def __call__(self, func, budget=1000):
        """
        Optimize the black box function using evolutionary algorithm.

        Parameters:
        func (callable): The black box function to optimize.
        budget (int, optional): The maximum number of function evaluations allowed. Defaults to 1000.

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
            evaluations = [func(self.population[i], budget) for i in range(pop_size)]

            # Select the fittest individuals
            self.population = self.select_fittest(pop_size, evaluations)

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i], budget) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < budget:
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

    def mutate(self, population, budget=1000):
        """
        Mutate the selected individuals.

        Parameters:
        population (np.ndarray): The selected individuals.
        budget (int, optional): The maximum number of function evaluations allowed. Defaults to 1000.

        Returns:
        np.ndarray: The mutated individuals.
        """
        # Create a copy of the population
        mutated = population.copy()

        # Randomly swap two individuals in the population
        for i in range(len(mutated)):
            j = np.random.choice(len(mutated))
            mutated[i], mutated[j] = mutated[j], mutated[i]

        # Refine the mutated individuals based on their performance
        for i in range(len(mutated)):
            if i < len(mutated) // 2:
                mutated[i] = self.refine(mutated[i], budget)
            else:
                mutated[i] = self.refine(mutated[i-1], budget)

        return mutated

    def refine(self, individual, budget):
        """
        Refine the individual based on its performance.

        Parameters:
        individual (float): The individual to refine.
        budget (int): The maximum number of function evaluations allowed.

        Returns:
        float: The refined individual.
        """
        # Initialize the population size and the number of generations
        pop_size = 100
        num_generations = 100

        # Initialize the population with the individual
        self.population = np.array([individual])

        # Run the evolutionary algorithm
        for gen in range(num_generations):
            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i], budget) for i in range(pop_size)]

            # Select the fittest individuals
            self.population = self.select_fittest(pop_size, evaluations)

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i], budget) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < budget:
                break

        # Return the refined individual
        return self.population[0]