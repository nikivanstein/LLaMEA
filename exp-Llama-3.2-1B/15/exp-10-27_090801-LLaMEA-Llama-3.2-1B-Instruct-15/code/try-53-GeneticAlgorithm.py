import numpy as np
import random
import math

class GeneticAlgorithm:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the genetic algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func, population_size=100, mutation_rate=0.01, crossover_rate=0.5):
        """
        Optimize the black box function `func` using a genetic algorithm.

        Args:
            func (callable): The black box function to optimize.
            population_size (int, optional): The size of the population. Defaults to 100.
            mutation_rate (float, optional): The rate at which individuals mutate. Defaults to 0.01.
            crossover_rate (float, optional): The rate at which individuals crossover. Defaults to 0.5.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the fitness of each individual in the population
        fitnesses = np.array([func(individual) for individual in population])

        # Select the fittest individuals
        fittest_indices = np.argsort(fitnesses)[:self.budget]
        fittest_individuals = population[fittest_indices]

        # Create a new population by mutating and crossing over the fittest individuals
        new_population = []
        while len(new_population) < population_size:
            # Select two parents from the fittest individuals
            parent1, parent2 = random.sample(fittest_individuals, 2)

            # Mutate the parents with a probability of mutation_rate
            if random.random() < mutation_rate:
                parent1 = parent1 + random.normal(0, 1, self.dim)

            # Cross over the parents with a probability of crossover_rate
            if random.random() < crossover_rate:
                child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
                child = np.random.uniform(-5.0, 5.0, child.shape)
                new_population.append(child)

            # Add the new individual to the new population
            new_population.append(parent1)

        # Evaluate the fitness of the new population
        fitnesses = np.array([func(individual) for individual in new_population])

        # Select the fittest individual in the new population
        fittest_indices = np.argsort(fitnesses)[:self.budget]
        fittest_individuals = new_population[fittest_indices]

        # Return the optimized parameter values and the objective function value
        return fittest_individuals, fitnesses[-1]

# Example usage:
def func(x):
    return np.sum(x**2)

ga = GeneticAlgorithm(budget=100, dim=10)
individuals = ga(__call__(func))

# Save the results to a file
np.save("currentexp/ga-aucs-0.npy", individuals)

# Print the results
print("Optimized parameter values:", individuals[0])
print("Objective function value:", individuals[0][-1])