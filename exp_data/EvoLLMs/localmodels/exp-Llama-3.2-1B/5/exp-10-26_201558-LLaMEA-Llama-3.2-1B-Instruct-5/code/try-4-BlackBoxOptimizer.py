# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(np.abs(func(population)))[:, :self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Evaluate the function for each individual in the new population
            new_func_values = func(new_population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(np.abs(new_func_values))[:, :self.population_size // 2]

            # Replace the old population with the new population
            population = new_population

            # Update the mutation probabilities
            self.updateMutationProbabilities(fittest_individuals, new_func_values)

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def updateMutationProbabilities(self, fittest_individuals, new_func_values):
        """
        Update the mutation probabilities based on the fittest individuals and new function values.

        Parameters:
        fittest_individuals (numpy array): The fittest individuals in the population.
        new_func_values (numpy array): The new function values.
        """
        # Calculate the mutation probabilities
        mutation_probabilities = np.random.uniform(0.01, 0.1, size=(self.population_size, self.dim))

        # Update the mutation probabilities
        self.mutation_probabilities = mutation_probabilities

        # Normalize the mutation probabilities
        self.mutation_probabilities /= np.sum(self.mutation_probabilities, axis=1, keepdims=True)

# One-line description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# BlackBoxOptimizer: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# ```