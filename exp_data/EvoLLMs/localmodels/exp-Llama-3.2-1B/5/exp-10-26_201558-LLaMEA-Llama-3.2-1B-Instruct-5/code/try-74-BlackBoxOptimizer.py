# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
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
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(np.abs(func(population)))[:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Update the mutation rate
            mutation_rate = self.updateMutationRate(population, func)

            # Apply the mutation
            for i in range(population_size):
                if random.random() < mutation_rate:
                    # Select a random individual
                    individual = population[i]

                    # Generate a new individual by swapping two random elements
                    swapped_individual = individual[:self.dim // 2] + [individual[self.dim // 2] + random.uniform(-5.0, 5.0) for _ in range(self.dim // 2)]

                    # Replace the individual in the population
                    population[i] = swapped_individual

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def updateMutationRate(self, population, func):
        """
        Update the mutation rate based on the function values.

        Parameters:
        population (numpy array): The current population.
        func (function): The black box function.

        Returns:
        float: The updated mutation rate.
        """
        # Calculate the average function value
        avg_func_value = np.mean(func(population))

        # Calculate the mutation rate
        mutation_rate = 0.05 + (0.1 * avg_func_value - 0.05)

        return mutation_rate

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

class GeneticProgramOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the GeneticProgramOptimizer with a given budget and dimensionality.

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
            fittest_individuals = np.argsort(np.abs(func(population)))[:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Apply the mutation
            for i in range(population_size):
                if random.random() < 0.5:  # 50% chance of mutation
                    # Select a random individual
                    individual = population[i]

                    # Generate a new individual by swapping two random elements
                    swapped_individual = individual[:self.dim // 2] + [individual[self.dim // 2] + random.uniform(-5.0, 5.0) for _ in range(self.dim // 2)]

                    # Replace the individual in the population
                    population[i] = swapped_individual

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

def fitness_func(population, func):
    """
    Evaluate the fitness of the population.

    Parameters:
    population (numpy array): The current population.
    func (function): The black box function.

    Returns:
    float: The fitness of the population.
    """
    # Evaluate the function for each individual in the population
    func_values = func(population)

    # Return the average function value
    return np.mean(func_values)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

def mutation(individual, func):
    """
    Apply a mutation to the individual.

    Parameters:
    individual (numpy array): The individual to mutate.
    func (function): The black box function.

    Returns:
    numpy array: The mutated individual.
    """
    # Generate a new individual by swapping two random elements
    swapped_individual = individual[:self.dim // 2] + [individual[self.dim // 2] + random.uniform(-5.0, 5.0) for _ in range(self.dim // 2)]

    return swapped_individual

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

def main():
    # Set the parameters
    budget = 1000
    dim = 10
    population_size = 100

    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(budget, dim)

    # Run the optimization
    optimized_params, optimized_func = optimizer(__call__)

    # Plot the results
    plt.plot([fitness_func(optimized_params, func) for func in [np.sin, np.cos, np.exp, np.log]])
    plt.xlabel('Number of function evaluations')
    plt.ylabel('Fitness')
    plt.title('Fitness over function evaluations')
    plt.show()

    # Plot the mutation rates
    mutation_rates = [mutation(optimized_params, func) for func in [np.sin, np.cos, np.exp, np.log]]
    plt.plot([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
    plt.xlabel('Number of function evaluations')
    plt.ylabel('Mutation rate')
    plt.title('Mutation rate over function evaluations')
    plt.show()

if __name__ == "__main__":
    main()