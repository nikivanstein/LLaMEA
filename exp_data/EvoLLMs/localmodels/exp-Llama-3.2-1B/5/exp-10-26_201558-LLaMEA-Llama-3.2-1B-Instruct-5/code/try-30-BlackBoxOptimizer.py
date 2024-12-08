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
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Evaluate the function for each individual in the new population
            new_func_values = np.array([func(individual) for individual in new_population])

            # Select the fittest individuals based on the new function values
            fittest_individuals_new = np.argsort(new_func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[fittest_individuals_new.size // 2:],
                                              fittest_individuals_new[:fittest_individuals_new.size // 2]])

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

def mutation(individual, budget):
    """
    Apply a novel mutation strategy to the individual.

    Parameters:
    individual (numpy array): The individual to mutate.
    budget (int): The maximum number of mutations allowed.

    Returns:
    tuple: The mutated individual and the number of mutations applied.
    """
    # Select a random index to mutate
    index = np.random.randint(0, individual.shape[0])

    # Apply the mutation
    individual[index] += np.random.uniform(-0.1, 0.1)

    # Limit the mutation to the search space
    if individual[index] < -5.0:
        individual[index] = -5.0
    elif individual[index] > 5.0:
        individual[index] = 5.0

    # Return the mutated individual and the number of mutations applied
    return individual, 1

def selection(population, budget):
    """
    Select the fittest individuals based on the function values.

    Parameters:
    population (numpy array): The population to select from.
    budget (int): The number of fittest individuals to select.

    Returns:
    tuple: The fittest individuals and the number of fittest individuals selected.
    """
    # Select the fittest individuals based on the function values
    fittest_individuals = np.argsort(population)[::-1][:budget]

    # Return the fittest individuals and the number of fittest individuals selected
    return fittest_individuals, budget

def main():
    # Create an instance of the BlackBoxOptimizer
    optimizer = BlackBoxOptimizer(budget=100, dim=10)

    # Optimize the function using the BlackBoxOptimizer
    optimized_parameters, optimized_function_value = optimizer(func, 1000)

    # Print the results
    print("Optimized Parameters:", optimized_parameters)
    print("Optimized Function Value:", optimized_function_value)

if __name__ == "__main__":
    main()