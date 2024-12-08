import numpy as np
import random
import os

class EvolutionaryStrategy:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the evolutionary strategy.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        """
        Optimize the black box function `func` using evolutionary strategy.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.population + self.noise * np.random.normal(0, 1, self.dim))

            # Update the population with the new parameter values
            self.population = [p + self.noise * np.random.normal(0, 1, self.dim) for p in self.population]

        # Return the optimized parameter values and the objective function value
        return self.population, func(func_value)

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (tuple): A tuple containing the optimized parameter values.

        Returns:
            float: The fitness value of the individual.
        """
        func_value = func(individual)
        return func_value

# Black Box Optimization Function
def func(x):
    return x[0]**2 + x[1]**2

# Initialize the evolutionary strategy
evolutionary_strategy = EvolutionaryStrategy(100, 2)

# Evaluate the fitness of the initial population
individuals, fitness_values = evolutionary_strategy(__call__, func)

# Update the evolutionary strategy
for i in range(100):
    # Select a parent using tournament selection
    parent = random.choice([individual for individual, fitness in zip(evolutionary_strategy.population, fitness_values) if fitness == max(fitness_values)])

    # Create a new individual by mutation
    new_individual = evolutionary_strategy.population[:]

    # Evaluate the fitness of the new individual
    fitness = evolutionary_strategy.evaluate_fitness(new_individual)

    # Update the population with the new individual
    new_individual = evolutionary_strategy.population[:]

    # Update the evolutionary strategy with the new individual
    evolutionary_strategy.population = [new_individual]
    evolutionary_strategy.population.append(parent)
    evolutionary_strategy.population = [p for p in evolutionary_strategy.population if p!= parent]
    evolutionary_strategy.population = [p for p in evolutionary_strategy.population if p!= new_individual]

# Print the final population and fitness values
print("Final Population:")
print(evolutionary_strategy.population)
print("Final Fitness Values:")
print([fitness for individual, fitness in zip(evolutionary_strategy.population, fitness_values)])

# Save the final population and fitness values to a file
np.save("currentexp/aucs-EvoStrat-0.npy", [fitness for individual, fitness in zip(evolutionary_strategy.population, fitness_values)])

# Print the final fitness values
print("Final Fitness Values:")
print([fitness for individual, fitness in zip(evolutionary_strategy.population, fitness_values)])