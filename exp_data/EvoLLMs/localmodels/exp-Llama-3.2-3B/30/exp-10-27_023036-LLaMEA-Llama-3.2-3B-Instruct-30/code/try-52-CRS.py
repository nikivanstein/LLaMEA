import numpy as np
import random

class CRS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random individuals
        population = []
        for _ in range(self.budget):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        # Evaluate the fitness of each individual in the population
        fitness = [func(individual) for individual in self.population]
        # Select the fittest individual
        fittest_individual = np.argmax(fitness)
        return fittest_individual

    def crossover(self, parent1, parent2):
        # Perform crossover to create a new individual
        child = parent1 + (parent2 - parent1) * np.random.uniform(0, 1)
        return child

    def recombination(self, parent1, parent2):
        # Perform recombination to create a new individual
        child = parent1 + (parent2 - parent1) * np.random.uniform(0, 1)
        return child

    def selection(self, population):
        # Select the fittest individual
        fitness = [func(individual) for individual in population]
        fittest_individual = np.argmax(fitness)
        return population[fittest_individual]

    def refine(self, selected_individual):
        # Refine the strategy of the selected individual with a probability of 0.3
        if random.random() < 0.3:
            # Randomly change a dimension of the individual
            dim_to_change = random.randint(0, self.dim - 1)
            self.population[self.population.index(selected_individual)][dim_to_change] += np.random.uniform(-1.0, 1.0)
        return selected_individual

    def optimize(self, func):
        # Optimize the black box function using the CRS algorithm
        for _ in range(self.budget):
            # Evaluate the fitness of each individual in the population
            fittest_individual = self.evaluate(func)
            # Select the fittest individual
            selected_individual = self.selection(self.population)
            # Refine the strategy of the selected individual
            selected_individual = self.refine(selected_individual)
            # Create a new individual by crossover and recombination
            new_individual = self.crossover(selected_individual, selected_individual)
            # Replace the fittest individual with the new individual
            self.population[fittest_individual] = new_individual

# Usage
def func(x):
    return np.sum(x**2)

crs = CRS(budget=100, dim=10)
crs.optimize(func)
print(crs.population)