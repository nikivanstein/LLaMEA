import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize population with random solutions
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Optimize the black box function using the current population
        def fitness(individual):
            func(individual)
            return individual

        # Evaluate the fitness of each individual
        fitness_values = [fitness(individual) for individual in self.population]

        # Select the best individuals based on their fitness values
        selected_individuals = []
        for _ in range(int(self.budget / 2)):
            max_index = fitness_values.index(max(fitness_values))
            selected_individuals.append(self.population[max_index])
            fitness_values[max_index] = -np.inf  # Assign negative fitness value to avoid selection pressure

        # Create new offspring by adapting the selected individuals
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < 0.5:
                child = parent1
            offspring.append(child)

        # Mutate the offspring with a small probability
        for individual in offspring:
            if random.random() < 0.05:
                index1, index2 = random.sample(range(self.dim), 2)
                individual[index1], individual[index2] = random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)

        # Replace the current population with the new offspring
        self.population = offspring

        # Evaluate the fitness of the new population
        fitness_values = [fitness(individual) for individual in self.population]
        selected_individuals = []
        for _ in range(int(self.budget / 2)):
            max_index = fitness_values.index(max(fitness_values))
            selected_individuals.append(self.population[max_index])
            fitness_values[max_index] = -np.inf  # Assign negative fitness value to avoid selection pressure

        # Select the best individuals based on their fitness values
        selected_individuals = []
        for _ in range(int(self.budget / 2)):
            max_index = fitness_values.index(max(fitness_values))
            selected_individuals.append(self.population[max_index])
            fitness_values[max_index] = -np.inf  # Assign negative fitness value to avoid selection pressure

        return selected_individuals

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        if random.random() < 0.05:
            index1, index2 = random.sample(range(self.dim), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual