import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
from collections import deque
import random

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float], mutation_prob: float) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        mutation_prob (float): The probability of applying a mutation to an individual.
        """
        self.budget = budget
        self.dim = dim
        self.func = func
        self.mutation_prob = mutation_prob
        self.population = self.generate_population()

    def generate_population(self) -> list:
        """
        Generate an initial population of random individuals.

        Returns:
        list: A list of dictionaries representing the initial population.
        """
        return [{k: np.random.uniform(-5.0, 5.0, self.dim) for k in self.func} for _ in range(100)]

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using a novel heuristic algorithm.

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

        # Use the minimize function to optimize the black box function
        result = minimize(objective, x, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda x: 0}, x0=x)

        # Return the optimized function values
        return {k: -v for k, v in result.x.items()}

    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        Mutate an individual with a probability of mutation.

        Args:
        individual (Dict[str, float]): A dictionary representing an individual.

        Returns:
        Dict[str, float]: The mutated individual.
        """
        if random.random() < self.mutation_prob:
            # Randomly select a variable to mutate
            var = random.choice(list(individual.keys()))

            # Generate a new value for the mutated variable
            new_val = random.uniform(-5.0, 5.0)

            # Update the individual with the new value
            individual[var] = new_val

        return individual

    def evolve_population(self, population: list, budget: int) -> list:
        """
        Evolve the population by applying the mutation operator.

        Args:
        population (list): A list of dictionaries representing the population.
        budget (int): The number of function evaluations allowed.

        Returns:
        list: The evolved population.
        """
        evolved_population = []

        for _ in range(budget):
            # Evaluate the fitness of each individual
            fitness = [self.__call__(func) for func in population]

            # Select the fittest individuals
            fittest_individuals = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)[:self.budget // 2]

            # Evolve the fittest individuals
            evolved_individuals = [self.mutate(individual) for individual, _ in fittest_individuals]

            # Add the evolved individuals to the evolved population
            evolved_population.extend(evolved_individuals)

        return evolved_population

# Description: BlackBoxOptimizer: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 