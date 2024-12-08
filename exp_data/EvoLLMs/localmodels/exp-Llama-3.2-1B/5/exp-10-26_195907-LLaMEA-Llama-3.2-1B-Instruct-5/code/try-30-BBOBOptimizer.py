# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        # Initialize population with random solutions
        population = self.initialize_population(self.budget)

        while True:
            # Select the fittest individual using tournament selection
            fittest_individuals = self.select_fittest(population, self.budget)

            # Refine the fittest individuals using tournament selection with probability 0.95
            new_individuals = self.tournament_selection(fittest_individuals, self.budget, self.search_space)

            # Evaluate the fitness of the new individuals
            new_population = self.evaluate_fitness(new_individuals, self.func)

            # Replace the old population with the new one
            population = new_population

            # Check if the population has reached the maximum size
            if len(population) >= self.budget:
                break

            # Refine the population using elitism
            population = self.elitism(population, self.budget, self.search_space)

        return population

    def initialize_population(self, budget):
        # Initialize the population with random solutions
        population = np.random.uniform(self.search_space, size=(budget, self.dim, self.dim))
        return population

    def select_fittest(self, population, budget):
        # Select the fittest individuals using tournament selection
        fittest_individuals = []
        for _ in range(budget):
            # Randomly select an individual
            individual = random.choice(population)

            # Evaluate the fitness of the individual
            fitness = self.func(individual)

            # Add the individual to the list of fittest individuals
            fittest_individuals.append((individual, fitness))

        # Return the fittest individuals
        return fittest_individuals

    def tournament_selection(self, fittest_individuals, budget, search_space):
        # Refine the fittest individuals using tournament selection with probability 0.95
        new_individuals = []
        for _ in range(budget):
            # Randomly select two individuals
            individual1, fitness1 = random.choice(fittest_individuals)
            individual2, fitness2 = random.choice(fittest_individuals)

            # Evaluate the fitness of the two individuals
            fitness = min(fitness1[1], fitness2[1])

            # Add the individual with the lower fitness to the list of new individuals
            if fitness == fitness1[1]:
                new_individuals.append(individual1)
            else:
                new_individuals.append(individual2)

        # Return the new individuals
        return new_individuals

    def elitism(self, population, budget, search_space):
        # Refine the population using elitism
        new_population = population
        for _ in range(budget - 1):
            # Select the fittest individual using tournament selection
            fittest_individual = random.choice(population)

            # Add the fittest individual to the new population
            new_population = np.vstack((new_population, [fittest_individual]))

        # Return the new population
        return new_population

# One-line description
# Novel Metaheuristic Algorithm for Black Box Optimization
# 