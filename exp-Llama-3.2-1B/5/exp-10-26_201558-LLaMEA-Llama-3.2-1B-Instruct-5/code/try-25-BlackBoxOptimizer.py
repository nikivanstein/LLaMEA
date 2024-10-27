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

            # Evaluate the new population to refine the strategy
            refined_population, refined_func_values = self.refine_strategy(population, new_population, func_values)

            # Replace the old population with the new population
            population = refined_population

        # Return the optimized parameters and the optimized function value
        return population, refined_func_values

    def refine_strategy(self, population, new_population, func_values):
        """
        Refine the strategy by applying a novel mutation strategy.

        Parameters:
        population (numpy array): The current population.
        new_population (numpy array): The new population.
        func_values (numpy array): The function values.

        Returns:
        tuple: The refined population and the refined function values.
        """
        # Define the mutation parameters
        mutation_rate = 0.01

        # Define the mutation operators
        def mutate(individual):
            # Select a random individual from the new population
            new_individual = new_population[np.random.choice(new_population.shape[0], 1)]

            # Apply the mutation operator to the individual
            mutated_individual = individual + np.random.normal(0, 1, new_individual.shape)

            # Clip the mutated individual to the search space
            mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

            return mutated_individual

        # Evaluate the function for each individual in the new population
        refined_func_values = np.array([func(new_individual) for new_individual in new_population])

        # Select the fittest individuals based on the function values
        fittest_individuals = np.argsort(refined_func_values)[::-1][:self.population_size // 2]

        # Create a new population by combining the fittest individuals
        refined_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        # Replace the old population with the new population
        population = refined_population

        return refined_population, refined_func_values