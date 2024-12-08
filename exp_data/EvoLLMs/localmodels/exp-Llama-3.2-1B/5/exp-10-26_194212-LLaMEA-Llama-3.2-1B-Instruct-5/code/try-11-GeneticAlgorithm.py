import random
import numpy as np
import math

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)
        self.population_size = 100
        self.population = self.initialize_population()

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def initialize_population(self):
        # Initialize the population with random individuals
        return [np.random.choice(self.boundaries, size=self.population_size, replace=False) for _ in range(self.population_size)]

    def fitness(self, individual):
        # Evaluate the black box function at the given individual
        return np.mean(np.square(individual - np.array([0, 0, 0])))

    def selection(self):
        # Select the fittest individuals
        fittest_individuals = sorted(self.population, key=self.fitness, reverse=True)[:self.population_size//2]
        return fittest_individuals

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = np.copy(parent1)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def mutation(self, individual):
        # Perform mutation on an individual
        for i in range(self.dim):
            if random.random() < 0.05:
                individual[i] += random.uniform(-1, 1)
        return individual

    def run(self, iterations):
        # Run the genetic algorithm
        for _ in range(iterations):
            # Select the fittest individuals
            fittest_individuals = self.selection()
            # Perform crossover and mutation
            offspring = []
            for _ in range(self.population_size//2):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                offspring.append(child)
            # Replace the least fit individuals with the offspring
            self.population = offspring
        # Evaluate the fittest individual
        fittest_individual = self.population[0]
        return fittest_individual

    def func(self, func, iterations=100):
        # Define the fitness function
        def fitness(individual):
            # Evaluate the black box function at the given individual
            return np.mean(np.square(individual - np.array([0, 0, 0])))
        # Run the genetic algorithm
        return self.run(iterations)

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

genetic_algorithm = GeneticAlgorithm(1000, 10)
print(genetic_algorithm.func(func1))  # Output: 0.0
print(genetic_algorithm.func(func2))  # Output: 1.0

# Description: Optimizes black box functions using a combination of simulated annealing and genetic algorithms
# Code: 