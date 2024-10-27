import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(min(self.budget, len(self.search_space))):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

def mutate(individual):
    """
    Mutate an individual by changing a random bit to 1.

    Args:
        individual (list): The individual to mutate.

    Returns:
        list: The mutated individual.
    """
    mutated_individual = individual.copy()
    mutated_individual[random.randint(0, len(individual) - 1)] = 1
    return mutated_individual

def crossover(parent1, parent2):
    """
    Perform crossover between two parents to generate a child.

    Args:
        parent1 (list): The first parent.
        parent2 (list): The second parent.

    Returns:
        list: The child.
    """
    child = parent1[:len(parent1) // 2] + parent2[len(parent2) // 2:]
    return child

def selection(population, num_individuals):
    """
    Select the best individuals from a population using tournament selection.

    Args:
        population (list): The population.
        num_individuals (int): The number of individuals to select.

    Returns:
        list: The selected individuals.
    """
    winners = []
    for _ in range(num_individuals):
        winner_index = np.random.randint(0, len(population))
        winner = population[winner_index]
        for other_index in range(len(population)):
            if other_index!= winner_index:
                other = population[other_index]
                if np.random.rand() < np.mean([winner[0] - other[0], winner[1] - other[1], winner[2] - other[2]]):
                    winner_index = other_index
        winners.append(winner)
    return winners

def genetic_algorithm(func, population_size, mutation_rate, crossover_rate, num_individuals, population_size_init, mutation_rate_init, crossover_rate_init):
    """
    Perform a genetic algorithm to optimize a black box function.

    Args:
        func (callable): The black box function to optimize.
        population_size (int): The initial population size.
        mutation_rate (float): The mutation rate.
        crossover_rate (float): The crossover rate.
        num_individuals (int): The number of individuals to select.
        population_size_init (int): The initial population size.
        mutation_rate_init (float): The initial mutation rate.
        crossover_rate_init (float): The initial crossover rate.

    Returns:
        tuple: The best individual and its fitness.
    """
    # Initialize the population
    population = np.random.uniform(-5.0, 5.0, (population_size, population_size))
    for _ in range(population_size_init):
        population = np.random.uniform(-5.0, 5.0, (population_size, population_size))

    # Initialize the best individual and its fitness
    best_individual = None
    best_fitness = float('-inf')

    # Run the genetic algorithm
    for _ in range(100):
        # Select the best individuals
        population = selection(population, num_individuals)

        # Perform crossover and mutation
        for i in range(num_individuals):
            parent1 = population[i]
            parent2 = population[(i + 1) % num_individuals]
            child = crossover(parent1, parent2)
            child = mutate(child)
            child = crossover(child, parent1)
            child = mutate(child)

            # Evaluate the fitness of the child
            child_fitness = func(child)
            if child_fitness > best_fitness:
                best_individual = child
                best_fitness = child_fitness

        # Update the best individual and its fitness
        if best_fitness > best_fitness_init:
            best_individual = best_individual
            best_fitness = best_fitness

    # Return the best individual and its fitness
    return best_individual, best_fitness

def optimize_func(func, budget, dim, population_size, mutation_rate, crossover_rate, num_individuals):
    """
    Optimize a black box function using a genetic algorithm.

    Args:
        func (callable): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        population_size (int): The population size.
        mutation_rate (float): The mutation rate.
        crossover_rate (float): The crossover rate.
        num_individuals (int): The number of individuals to select.

    Returns:
        tuple: The best individual and its fitness.
    """
    # Initialize the genetic algorithm
    best_individual, best_fitness = genetic_algorithm(func, population_size, mutation_rate, crossover_rate, num_individuals, population_size_init, mutation_rate_init, crossover_rate_init)

    # Return the best individual and its fitness
    return best_individual, best_fitness

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 1000
dim = 5
population_size = 100
mutation_rate = 0.1
crossover_rate = 0.5
num_individuals = 100

best_individual, best_fitness = optimize_func(func, budget, dim, population_size, mutation_rate, crossover_rate, num_individuals)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)