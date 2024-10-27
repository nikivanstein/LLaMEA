import random
import numpy as np
import copy

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
        for _ in range(self.budget):
            # Generate a random point in the search space
            new_individual = copy.deepcopy(self.search_space[np.random.randint(0, self.dim)])

            # Evaluate the function at the current point
            value = func(new_individual)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = new_individual

        # Return the optimized value
        return best_value

class GeneticAlgorithm(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        """
        Initialize the GeneticAlgorithm with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        super().__init__(budget, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the GeneticAlgorithm.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population
        population = self.generate_population(self.budget)

        # Evaluate the fitness of each individual in the population
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]

        # Select the fittest individuals
        fittest_individuals = self.select_fittest(population, fitnesses)

        # Perform crossover and mutation
        offspring = self.crossover_and_mutate(fittest_individuals, fitnesses)

        # Evaluate the fitness of the new population
        new_fitnesses = [self.evaluate_fitness(individual, func) for individual in offspring]

        # Select the fittest individuals in the new population
        new_fittest_individuals = self.select_fittest(offspring, new_fitnesses)

        # Return the optimized value
        return self.evaluate_fitness(new_fittest_individuals[0], func)

def generate_population(budget):
    """
    Generate a population of individuals using the roulette wheel selection.

    Args:
        budget (int): The maximum number of function evaluations allowed.

    Returns:
        list: A list of individuals in the population.
    """
    population = []
    for _ in range(budget):
        # Generate a random point in the search space
        point = np.random.choice(self.search_space, size=self.dim)

        # Evaluate the function at the current point
        value = func(point)

        # Add the individual to the population
        population.append(point)

    return population

def select_fittest(population, fitnesses):
    """
    Select the fittest individuals in the population based on their fitness.

    Args:
        population (list): A list of individuals in the population.
        fitnesses (list): A list of fitness values corresponding to each individual.

    Returns:
        list: A list of fittest individuals in the population.
    """
    # Calculate the fitness scores
    scores = [fitness / len(fitnesses) for fitness in fitnesses]

    # Select the fittest individuals based on their fitness scores
    fittest_individuals = [individual for _, individual in sorted(zip(scores, population), reverse=True)]

    return fittest_individuals

def crossover_and_mutate(parent1, parent2):
    """
    Perform crossover and mutation on two parent individuals.

    Args:
        parent1 (list): The first parent individual.
        parent2 (list): The second parent individual.

    Returns:
        list: A list of offspring individuals.
    """
    # Perform crossover
    offspring = []
    for _ in range(len(parent1) // 2):
        offspring.append(parent1[np.random.randint(0, len(parent1))])
        offspring.append(parent2[np.random.randint(0, len(parent2))])

    # Perform mutation
    for individual in offspring:
        # Generate a random mutation point
        mutation_point = np.random.randint(0, len(individual))

        # Swap the two points
        individual[mutation_point], individual[mutation_point + 1] = individual[mutation_point + 1], individual[mutation_point]

    return offspring

def evaluate_fitness(individual, func):
    """
    Evaluate the fitness of an individual in the population.

    Args:
        individual (list): The individual to evaluate.
        func (callable): The black box function to evaluate.

    Returns:
        float: The fitness value of the individual.
    """
    return func(individual)

# Example usage
def func(x):
    return x**2 + 2*x + 1

ga = GeneticAlgorithm(100, 5)
optimized_value = ga(func)
print("Optimized value:", optimized_value)