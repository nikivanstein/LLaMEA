import numpy as np
import random
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a random solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the solution
            fitness = func(solution)
            # Store the fitness score and solution
            self.population.append((fitness, solution))
            self.fitness_scores.append(fitness)

        # Select the best individual based on the fitness scores
        self.best_individual = self.population[np.argmin(self.fitness_scores)]

    def mutate(self, individual):
        # Randomly modify the individual within the search space
        mutated_individual = individual + np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(1, self.dim - 1)
        # Create a child individual by combining the two parents
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def select_parents(self, num_parents):
        # Select parents based on their fitness scores
        parents = []
        while len(parents) < num_parents:
            fitness, individual = self.population[np.argmin(self.fitness_scores)]
            parents.append((fitness, individual))
        return parents

    def evolve(self):
        # Evolve the population using mutation and crossover
        while len(self.population) > 0:
            # Select the best individual based on its fitness score
            best_individual = self.population[np.argmin(self.fitness_scores)]
            # Select parents based on their fitness scores
            parents = self.select_parents(1)
            # Create a new population by combining the best individual and its parents
            new_population = []
            for _ in range(self.budget):
                # Select a random parent
                parent1, parent2 = random.sample(parents, 2)
                # Crossover the parents to create a child
                child = self.crossover(parent1, parent2)
                # Mutate the child
                child = self.mutate(child)
                # Add the child to the new population
                new_population.append(child)
            # Replace the old population with the new population
            self.population = new_population
            # Update the fitness scores
            self.fitness_scores = []
            for fitness, individual in self.population:
                self.fitness_scores.append(fitness)

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 