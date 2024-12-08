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
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Calculate the mutation probability based on the number of function evaluations
            mutation_prob = np.random.rand(self.population_size) / self.budget

            # Apply the mutation strategy
            for i in range(population_size):
                if np.random.rand() < mutation_prob:
                    # Select a random individual from the population
                    individual = population[i]

                    # Calculate the new individual using the mutation strategy
                    new_individual = np.concatenate([individual[:fittest_individuals.size // 2], np.random.uniform(fittest_individuals.size // 2, self.dim), fittest_individuals[fittest_individuals.size // 2:]])

                    # Replace the old individual with the new individual
                    population[i] = new_individual

        # Return the optimized parameters and the optimized function value
        return population, func(population)