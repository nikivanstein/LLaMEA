import numpy as np
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Generate initial population
        for _ in range(self.population_size):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            self.population.append(individual)

        # Evolve population for a specified number of generations
        for _ in range(1000):  # max iterations
            # Evaluate function for each individual
            fitness_scores = []
            for individual in self.population:
                func_value = func(individual)
                fitness_scores.append(func_value)
            fitness_scores = np.array(fitness_scores)

            # Select parents based on fitness scores
            parents = self.select_parents(fitness_scores, self.population_size)

            # Create offspring by crossover and mutation
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                offspring.append(child)

            # Replace old population with new one
            self.population = offspring

            # Update fitness scores
            fitness_scores = []
            for individual in self.population:
                func_value = func(individual)
                fitness_scores.append(func_value)
            fitness_scores = np.array(fitness_scores)

            # Select top individuals based on fitness scores
            parents = self.select_parents(fitness_scores, self.population_size)

            # Replace old population with new one
            self.population = parents

        # Return the fittest individual
        best_individual = self.population[np.argmax(self.fitness_scores)]
        return best_individual

    def select_parents(self, fitness_scores, population_size):
        # Select parents based on fitness scores
        # Use probability of 0.05 to refine strategy
        parents = []
        for _ in range(population_size):
            parent = random.choices(range(self.population_size), weights=fitness_scores, k=1)[0]
            parents.append(parent)
        return parents

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = parent1[:len(parent1)//2] + parent2[len(parent1)//2:]
        return child

    def mutate(self, individual):
        # Perform mutation on an individual
        # Use probability of 0.05 to refine strategy
        if random.random() < 0.05:
            index = random.randint(0, self.dim-1)
            individual[index] = random.uniform(-5.0, 5.0)
        return individual

# Description: Evolutionary Optimization using Genetic Programming
# Code: 