import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_initial_population()
        self.fitness_scores = self.calculate_fitness_scores(self.population)

    def generate_initial_population(self):
        # Generate an initial population with random solutions
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(100)]

    def calculate_fitness_scores(self, population):
        # Calculate fitness scores for each individual in the population
        fitness_scores = []
        for individual in population:
            func = lambda x: np.sum(np.abs(x - self.budget * x))
            fitness = func(individual)
            fitness_scores.append((fitness, individual))
        return fitness_scores

    def __call__(self, func):
        # Optimize the black box function using the Genetic Algorithm
        while True:
            # Select the fittest individuals
            fittest_individuals = sorted(self.population, key=lambda x: self.fitness_scores[-1][0], reverse=True)[:self.budget]
            
            # Create a new generation of individuals
            new_population = []
            for _ in range(self.budget):
                # Perform crossover and mutation on the fittest individuals
                child = []
                for _ in range(self.dim):
                    parent1, parent2 = random.sample(fittest_individuals, 2)
                    child.append(self.crossover(parent1, parent2))
                    child.append(self.mutation(child[-1]))
                new_population.append(child)
            
            # Replace the old population with the new one
            self.population = new_population
            self.fitness_scores = self.calculate_fitness_scores(self.population)
            if self.fitness_scores[-1][0] > self.fitness_scores[-2][0]:
                break

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = parent1[:parent1.index(max(parent1))] + parent2[parent2.index(max(parent2)):]
        return child

    def mutation(self, individual):
        # Perform mutation on an individual
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                mutated_individual[i] += random.uniform(-1, 1)
        return mutated_individual

# Description: Genetic Algorithm with Adaptive Crossover and Mutation
# Code: 