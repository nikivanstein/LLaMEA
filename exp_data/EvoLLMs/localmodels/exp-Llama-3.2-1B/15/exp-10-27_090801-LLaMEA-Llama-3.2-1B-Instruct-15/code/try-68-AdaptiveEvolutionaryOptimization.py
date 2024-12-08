import numpy as np
import random

class AdaptiveEvolutionaryOptimization:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the adaptive evolutionary optimization algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.population_size = 50
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize the population with random parameter values within the search space.

        Returns:
            list: A list of dictionaries containing the parameter values and their fitness values.
        """
        return [{"param_values": np.random.uniform(-5.0, 5.0, self.dim),
                "fitness": np.random.uniform(0, 100, self.dim)} for _ in range(self.population_size)]

    def __call__(self, func):
        """
        Optimize the black box function `func` using adaptive evolutionary optimization.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = self.initialize_population()

        # Evaluate the fitness of each individual in the population
        for individual in self.population:
            func_value = func(individual["param_values"])
            individual["fitness"] = func_value

        # Select the fittest individuals based on their fitness values
        self.population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)

        # Select the next generation based on the selected individuals
        next_generation = self.select_next_generation(self.population)

        # Update the population with the selected individuals
        self.population = next_generation

        # Update the noise level based on the number of evaluations
        self.noise = 0.1 * (1 - self.noise / self.budget)

        # Return the optimized parameter values and the objective function value
        return self.population[0]["param_values"], self.population[0]["fitness"]

    def select_next_generation(self, population):
        """
        Select the next generation of individuals based on the selected individuals.

        Args:
            population (list): A list of dictionaries containing the parameter values and their fitness values.

        Returns:
            list: A list of dictionaries containing the optimized parameter values and their fitness values.
        """
        next_generation = []
        for _ in range(self.population_size // 2):
            # Select two individuals from the population with a probability of 0.5
            individual1 = random.choice(population)
            individual2 = random.choice(population)

            # Select the fittest individual as the parent
            parent1 = max(individual1["param_values"], key=individual1["fitness"])
            parent2 = max(individual2["param_values"], key=individual2["fitness"])

            # Create a new individual by combining the parents
            child = {**parent1, **parent2}

            # Update the noise level based on the number of evaluations
            self.noise = 0.1 * (1 - self.noise / self.budget)

            # Evaluate the fitness of the child individual
            child["fitness"] = func(child["param_values"])

            # Add the child individual to the next generation
            next_generation.append(child)

        return next_generation

# One-line description with the main idea:
# Adaptive Evolutionary Optimization (AEO) is a novel metaheuristic algorithm that adapts the selection strategy based on the number of function evaluations, improving its efficiency and effectiveness in solving black box optimization problems.