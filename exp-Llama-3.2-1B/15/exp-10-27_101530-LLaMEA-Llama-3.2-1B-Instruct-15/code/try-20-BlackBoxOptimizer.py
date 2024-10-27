# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
import math
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.population_size = 100

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Generate a new individual by changing one random gene
        new_individual = copy.deepcopy(individual)
        if random.random() < self.mutation_rate:
            new_individual[math.floor(random.random() * self.dim)] = random.uniform(-5.0, 5.0)
        return new_individual

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = copy.deepcopy(parent1)
        if random.random() < self.crossover_rate:
            # Select a random gene from parent1 and parent2
            idx = random.randint(0, self.dim - 1)
            child[idx] = parent2[idx]
        return child

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        func_value = self.func(individual)
        return func_value

    def selection(self, population):
        # Select the fittest individuals
        selected_population = sorted(population, key=self.evaluate_fitness, reverse=True)
        return selected_population[:self.population_size]

    def run(self, func, population_size):
        # Run the algorithm
        population = [self.evaluate_fitness(individual) for individual in population_size]
        best_individual = max(population)
        self.best_fitness = self.evaluate_fitness(best_individual)
        self.best_individual = best_individual
        while self.func_evaluations < self.budget:
            # Select the fittest individuals
            population = self.selection(population)
            # Perform crossover and mutation
            for i in range(0, len(population), 2):
                parent1, parent2 = population[i], population[i + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                # Evaluate the fitness of the child
                func_value = self.evaluate_fitness(child)
                # Replace the parent with the child
                population[i], population[i + 1] = child, parent1
            # Update the best individual
            best_individual = max(population)
            self.best_individual = best_individual
            # Update the best fitness
            self.best_fitness = self.evaluate_fitness(best_individual)
            # Check if the budget is reached
            if self.func_evaluations == self.budget:
                return best_individual
        return self.best_individual

import numpy as np
import random

# Test the algorithm
func = lambda x: x**2
optimizer = BlackBoxOptimizer(100, 10)
best_individual = optimizer.run(func, 100)
print("Best individual:", best_individual)
print("Best fitness:", optimizer.evaluate_fitness(best_individual))