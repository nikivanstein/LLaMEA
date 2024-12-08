import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float]) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        """
        self.budget = budget
        self.dim = dim
        self.func = func

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using an adaptive genetic algorithm.

        Args:
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.

        Returns:
        Dict[str, float]: The optimized function values.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the objective function to minimize (negative of the original function)
        def objective(x: np.ndarray) -> float:
            return -np.sum(self.func.values(x))

        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Initialize the population size and the number of generations
        population_size = 100
        num_generations = 100

        # Initialize the population with random individuals
        population = self.initialize_population(population_size, self.dim)

        # Run the genetic algorithm for the specified number of generations
        for _ in range(num_generations):
            # Evaluate the fitness of each individual in the population
            fitnesses = [self.evaluate_fitness(individual, func, population) for individual in population]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitnesses)[-self.budget:]

            # Create a new population by crossover and mutation
            new_population = self.create_new_population(population, fittest_individuals, population_size, num_generations)

            # Replace the old population with the new one
            population = new_population

        # Return the optimized function values
        return {k: -v for k, v in self.evaluate_fitness(new_population[0], func, population)[0]}

    def initialize_population(self, population_size: int, dim: int) -> np.ndarray:
        """
        Initialize the population with random individuals.

        Args:
        population_size (int): The size of the population.
        dim (int): The dimensionality of the search space.

        Returns:
        np.ndarray: The initialized population.
        """
        return np.random.uniform(-5.0, 5.0, (population_size, dim)).astype(np.float64)

    def create_new_population(self, population: np.ndarray, fittest_individuals: np.ndarray, population_size: int, num_generations: int) -> np.ndarray:
        """
        Create a new population by crossover and mutation.

        Args:
        population (np.ndarray): The current population.
        fittest_individuals (np.ndarray): The fittest individuals.
        population_size (int): The size of the new population.
        num_generations (int): The number of generations.

        Returns:
        np.ndarray: The new population.
        """
        # Select the fittest individuals
        parents = population[fittest_individuals]

        # Perform crossover and mutation
        offspring = []
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(fittest_individuals, 2, replace=False)
            child = self.crossover(parent1, parent2)
            child = self.mutation(child)
            offspring.append(child)

        # Replace the old population with the new one
        population[:] = offspring

        return offspring

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents.

        Args:
        parent1 (np.ndarray): The first parent.
        parent2 (np.ndarray): The second parent.

        Returns:
        np.ndarray: The offspring.
        """
        # Select a random crossover point
        crossover_point = np.random.randint(0, parent1.shape[0])

        # Perform crossover
        child = parent1[:crossover_point] + parent2[crossover_point:]

        return child

    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Perform mutation on an individual.

        Args:
        individual (np.ndarray): The individual.

        Returns:
        np.ndarray: The mutated individual.
        """
        # Select a random mutation point
        mutation_point = np.random.randint(0, individual.shape[0])

        # Perform mutation
        individual[mutation_point] += np.random.uniform(-1, 1)

        return individual

# Description: Adaptive Black Box Optimization using Genetic Algorithm
# Code: 