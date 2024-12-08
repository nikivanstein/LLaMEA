import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.ratio = 0.15
        self.iterations = 0
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

    def select(self, population):
        # Select the fittest individuals based on their fitness
        fitnesses = [individual.fitness for individual in population]
        selected_individuals = sorted(population, key=lambda individual: fitnesses[individual.index(max(fitnesses))] if fitnesses else float('-inf'), reverse=True)[:self.population_size]
        return selected_individuals

    def mutate(self, population):
        # Randomly swap two individuals in the population
        for i in range(len(population) - 1):
            j = random.randint(i + 1, len(population) - 1)
            population[i], population[j] = population[j], population[i]
        return population

    def crossover(self, parent1, parent2):
        # Perform crossover to create a new individual
        child = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
        return child

    def evolve(self, population):
        # Evolve the population using the selected strategy
        selected_individuals = self.select(population)
        mutated_individuals = selected_individuals + self.population_size - selected_individuals
        mutated_population = self.mutate(mutated_individuals)
        return mutated_population

    def train(self, population, func):
        # Train the algorithm using the given function
        while True:
            population = self.evolve(population)
            new_individuals = self.select(population)
            for individual in new_individuals:
                fitness = individual.fitness
                updated_individual = func(individual)
                updated_individual.fitness = fitness
                self.population_size -= 1
                if self.population_size == 0:
                    break
            self.iterations += 1
            if self.iterations % 100 == 0:
                print(f"Iteration {self.iterations}: {self.population_size} individuals left")

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 