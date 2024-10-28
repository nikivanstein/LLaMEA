# Description: Black Box Optimization using Evolutionary Algorithm
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
            indices = np.argsort([mean - std * i for i in range(pop_size)])

            # Refine the selected individuals based on their performance
            new_individuals = []
            for i in range(pop_size):
                new_individual = self.evaluate_fitness(self.population[i])
                new_individuals.append(new_individual)
                if new_individual < evaluations[i]:
                    self.population[i] = new_individual

            # Mutate the selected individuals
            self.population = self.mutate(self.population)

            # Evaluate the function at each individual in the population
            evaluations = [func(self.population[i]) for i in range(pop_size)]

            # Check if the population has reached the budget
            if len(evaluations) < self.budget:
                break

        # Return the optimized parameters and the optimized function value
        return self.population, evaluations[-1]

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Parameters:
        individual (RealSingleObjectiveProblem): The individual to evaluate.

        Returns:
        float: The fitness of the individual.
        """
        # Calculate the mean and standard deviation of the function values
        mean = np.mean(individual.values)
        std = np.std(individual.values)

        # Refine the individual based on its performance
        refined_individual = individual.copy()
        refined_individual.values = refined_individual.values - std * refined_individual.values / mean

        return refined_individual.values[-1]

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

# Initialize the Black Box Optimizer
optimizer = BlackBoxOptimizer(1000, 5)

# Optimize the black box function using the optimizer
optimized_individual, optimized_function_value = optimizer(func)

# Print the optimized parameters and the optimized function value
print("Optimized Individual:", optimized_individual)
print("Optimized Function Value:", optimized_function_value)