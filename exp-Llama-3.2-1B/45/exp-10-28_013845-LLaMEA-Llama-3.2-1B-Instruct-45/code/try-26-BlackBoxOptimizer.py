import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
import random

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float], mutation_rate: float, n_generations: int) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, a black box function, mutation rate, and number of generations.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        mutation_rate (float): The probability of mutating an individual.
        n_generations (int): The number of generations to run the algorithm.
        """
        self.budget = budget
        self.dim = dim
        self.func = func
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.population = self.generate_population()

    def generate_population(self) -> List[Any]:
        """
        Generate a population of individuals using the given mutation rate.

        Returns:
        List[Any]: A list of individuals, where each individual is a dictionary representing a solution.
        """
        population = []
        for _ in range(100):  # Run the algorithm for 100 generations
            individual = self.generate_individual()
            population.append(individual)
        return population

    def generate_individual(self) -> Dict[str, float]:
        """
        Generate an individual solution by sampling from the given search space.

        Returns:
        Dict[str, float]: A dictionary representing the individual solution.
        """
        x = np.random.uniform(-5.0, 5.0, self.dim)
        return {k: random.uniform(-5.0, 5.0) for k in x}

    def evaluate_fitness(self, individual: Dict[str, float]) -> float:
        """
        Evaluate the fitness of an individual solution.

        Args:
        individual (Dict[str, float]): A dictionary representing the individual solution.

        Returns:
        float: The fitness of the individual solution.
        """
        return -np.sum(self.func.values(individual))

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using the adaptive black box optimization algorithm.

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

        # Run the algorithm for the given number of generations
        for _ in range(self.n_generations):
            # Select parents using tournament selection
            parents = []
            for _ in range(100):  # Run the tournament for 100 generations
                tournament_size = random.randint(2, 10)
                tournament_indices = random.sample(range(len(self.population)), tournament_size)
                tournament_results = [self.evaluate_fitness(individual) for individual in tournament_indices]
                tournament_winner = tournament_results.index(max(tournament_results))
                parents.append(self.population[tournament_winner])
            # Select parents using roulette wheel selection
            parents = random.choices(self.population, weights=[1 / len(self.population) for _ in range(len(self.population))], k=100)

            # Crossover (mate) two parents to create a new individual
            new_individual = {}
            for key, value in parents[0].items():
                if value < self.func.values(x):
                    new_individual[key] = value
                else:
                    new_individual[key] = x[key]
            # Mutate the new individual with the given mutation rate
            for key, value in new_individual.items():
                if random.random() < self.mutation_rate:
                    new_individual[key] += random.uniform(-1, 1)

            # Evaluate the fitness of the new individual
            fitness = self.evaluate_fitness(new_individual)

            # Replace the worst individual with the new individual
            worst_index = np.argmin([self.evaluate_fitness(individual) for individual in self.population])
            self.population[worst_index] = new_individual

        # Return the optimized function values
        return {k: -v for k, v in new_individual.items()}

# Description: Adaptive Black Box Optimization using Genetic Algorithm
# Code: 