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

    def __call__(self, func, budget=1000):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed. Defaults to 1000.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(budget):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func(population))[:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Update the mutation rate
            mutation_rate = 0.05
            if random.random() < mutation_rate:
                # Select a random individual and mutate it
                individual = population[np.random.choice(population_size)]
                mutated_individual = individual + random.uniform(-1, 1)
                # Clip the mutated individual to the search space
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
                # Replace the mutated individual with the new individual
                population[np.random.choice(population_size), :] = mutated_individual

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
def fitness_function(func, population):
    """
    Evaluate the fitness of the population using the given function.

    Parameters:
    func (function): The black box function to evaluate.
    population (numpy array): The population of individuals to evaluate.

    Returns:
    float: The fitness of the population.
    """
    # Evaluate the function for each individual in the population
    fitness_values = func(population)
    return np.mean(fitness_values)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
def mutation(individual):
    """
    Apply a mutation to the individual.

    Parameters:
    individual (numpy array): The individual to mutate.

    Returns:
    numpy array: The mutated individual.
    """
    # Select a random index
    index = random.randint(0, individual.size - 1)

    # Swap the elements at the selected index
    individual[index], individual[index + 1] = individual[index + 1], individual[index]

    return individual

def selection(population):
    """
    Select the fittest individuals based on the fitness values.

    Parameters:
    population (numpy array): The population of individuals.

    Returns:
    numpy array: The fittest individuals.
    """
    # Sort the population based on the fitness values
    sorted_indices = np.argsort(population)

    # Select the fittest individuals
    fittest_indices = sorted_indices[:len(population) // 2]

    # Create a new population by combining the fittest individuals
    new_population = np.concatenate([population[fittest_indices], population[:fittest_indices[-1]]])

    return new_population

def main():
    # Initialize the BlackBoxOptimizer
    optimizer = BlackBoxOptimizer(budget=1000, dim=10)

    # Optimize the function
    optimized_parameters, optimized_function_value = optimizer(func, budget=1000)

    # Print the optimized parameters and the optimized function value
    print("Optimized Parameters:", optimized_parameters)
    print("Optimized Function Value:", optimized_function_value)

if __name__ == "__main__":
    main()