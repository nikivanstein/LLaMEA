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

            # Apply novel mutation strategy
            for _ in range(int(population_size // 2)):
                # Select two random individuals
                individual1, individual2 = random.sample(fittest_individuals, 2)

                # Apply mutation to individual1
                mutated_individual1 = individual1 + random.uniform(-1, 1)

                # Check if the mutated individual is within the search space
                if -5.0 <= mutated_individual1 <= 5.0:
                    # Replace the mutated individual with the new individual
                    fittest_individuals.remove(individual1)
                    fittest_individuals.append(mutated_individual1)

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

def evaluate_fitness(individual, func, budget, logger):
    """
    Evaluate the fitness of an individual in the population.

    Parameters:
    individual (numpy array): The individual to evaluate.
    func (function): The black box function to evaluate.
    budget (int): The maximum number of function evaluations allowed.
    logger (object): The logger to use for logging.

    Returns:
    float: The fitness value of the individual.
    """
    # Evaluate the function for the individual
    func_values = func(individual)

    # Log the fitness value
    logger.info(f"Fitness value: {func_values}")

    # Check if the budget has been exceeded
    if len(func_values) >= budget:
        # Log the error message
        logger.error("Function evaluations exceeded the budget.")

    # Return the fitness value
    return func_values

# Example usage
def test_function(x):
    return x**2 + 2*x + 1

budget = 100
dim = 2

optimizer = BlackBoxOptimizer(budget, dim)
individual = np.random.uniform(-5.0, 5.0, (1, dim))
fitness = evaluate_fitness(individual, test_function, budget, None)
print(fitness)

optimizer = BlackBoxOptimizer(budget, dim)
individual = np.random.uniform(-5.0, 5.0, (1, dim))
fitness = evaluate_fitness(individual, test_function, budget, None)
print(fitness)

# Novel mutation strategy
def mutate_individual(individual):
    """
    Apply a novel mutation to an individual.

    Parameters:
    individual (numpy array): The individual to mutate.

    Returns:
    numpy array: The mutated individual.
    """
    # Select two random individuals
    individual1, individual2 = random.sample(individual, 2)

    # Apply mutation to individual1
    mutated_individual1 = individual1 + random.uniform(-1, 1)

    # Check if the mutated individual is within the search space
    if -5.0 <= mutated_individual1 <= 5.0:
        # Replace the mutated individual with the new individual
        individual = np.concatenate([individual1, mutated_individual1])

    # Return the mutated individual
    return individual

# Evaluate the fitness of the mutated individual
fitness = evaluate_fitness(mutate_individual(individual), test_function, budget, None)
print(fitness)