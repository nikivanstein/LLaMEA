import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def optimize(self, func):
        """
        Optimize the black box function using a genetic algorithm.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            dict: A dictionary containing the best individual, its fitness, and the mutation rate.
        """
        # Initialize the population size and mutation rate
        population_size = 100
        mutation_rate = 0.01

        # Initialize the population
        population = self.generate_population(population_size, self.dim)

        # Evolve the population
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.fitness, reverse=True)[:self.population_size // 2]

            # Perform crossover and mutation
            new_population = []
            for _ in range(population_size // 2):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

        # Evaluate the best individual
        best_individual = self.generate_individual(self.search_space, population)
        best_value = self.f(best_individual)

        # Return the best individual, its fitness, and the mutation rate
        return {
            "best_individual": best_individual,
            "best_value": best_value,
            "mutation_rate": mutation_rate
        }

    def generate_population(self, population_size, dim):
        """
        Generate a population of individuals.

        Args:
            population_size (int): The number of individuals in the population.
            dim (int): The dimensionality of the search space.

        Returns:
            list: A list of individuals.
        """
        return [[random.uniform(-5.0, 5.0) for _ in range(dim)] for _ in range(population_size)]

    def generate_individual(self, search_space, population):
        """
        Generate an individual.

        Args:
            search_space (list): The search space.
            population (list): The population.

        Returns:
            list: An individual.
        """
        return random.choice(population)

    def fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (list): The individual.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the function at the current individual
        value = self.func(individual)

        # Return the fitness
        return value

    def crossover(self, parent1, parent2):
        """
        Perform crossover.

        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.

        Returns:
            list: The child.
        """
        # Select a random crossover point
        crossover_point = np.random.randint(0, len(parent1))

        # Create the child
        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Return the child
        return child

    def mutate(self, individual, mutation_rate):
        """
        Perform mutation.

        Args:
            individual (list): The individual.
            mutation_rate (float): The mutation rate.

        Returns:
            list: The mutated individual.
        """
        # Select a random point to mutate
        mutation_point = np.random.randint(0, len(individual))

        # Perform mutation
        mutated_individual = individual.copy()
        mutated_individual[mutation_point] += np.random.uniform(-mutation_rate, mutation_rate)

        # Return the mutated individual
        return mutated_individual