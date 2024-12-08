import numpy as np
import random

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

            # Select a subset of individuals for mutation
            mutation_subset = np.random.choice(fittest_individuals, size=self.population_size // 2, replace=False)

            # Perform mutation on the selected individuals
            mutated_population = np.concatenate([population, mutation_subset])

            # Replace the old population with the mutated population
            population = mutated_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

class GeneticProgrammer:
    def __init__(self, budget, dim):
        """
        Initialize the GeneticProgrammer with a given budget and dimensionality.

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

            # Select a subset of individuals for mutation
            mutation_subset = np.random.choice(fittest_individuals, size=self.population_size // 2, replace=False)

            # Perform mutation on the selected individuals
            mutated_population = np.concatenate([population, mutation_subset])

            # Replace the old population with the mutated population
            population = mutated_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

def mutation(individual, mutation_rate):
    """
    Perform a mutation on the given individual.

    Parameters:
    individual (numpy array): The individual to mutate.
    mutation_rate (float): The probability of mutation.

    Returns:
    numpy array: The mutated individual.
    """
    # Select a random index for mutation
    index = np.random.randint(0, individual.shape[0])

    # Perform mutation
    individual[index] += np.random.uniform(-1, 1)

    # Ensure the mutated individual is within the search space
    individual[index] = np.clip(individual[index], -5.0, 5.0)

    return individual

def main():
    # Initialize the BlackBoxOptimizer and GeneticProgrammer
    optimizer = BlackBoxOptimizer(1000, 10)
    programmer = GeneticProgrammer(1000, 10)

    # Optimize the function
    func = lambda x: x**2
    optimized_params, optimized_func_value = optimizer(func)

    # Optimize the function again with a higher budget
    budget = 2000
    optimized_params, optimized_func_value = programmer(func)

    # Print the results
    print("Optimized Parameters:", optimized_params)
    print("Optimized Function Value:", optimized_func_value)

if __name__ == "__main__":
    main()