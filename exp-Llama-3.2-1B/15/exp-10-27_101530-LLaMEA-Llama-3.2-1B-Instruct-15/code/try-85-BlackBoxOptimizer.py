import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

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

    def novel_metaheuristic(self, func, mutation_rate, cooling_rate):
        # Initialize the population with random points in the search space
        population = [self.search_space[0] + random.uniform(-self.search_space[1], self.search_space[1]) for _ in range(100)]

        # Evaluate the fitness of each individual in the population
        fitness = [self.evaluate_fitness(individual, func) for individual in population]

        # Select the fittest individuals to reproduce
        parents = self.select_parents(fitness, population, self.budget)

        # Create a new population by crossover and mutation
        new_population = self.crossover(parents, mutation_rate, cooling_rate)

        # Evaluate the fitness of the new population
        new_fitness = [self.evaluate_fitness(individual, func) for individual in new_population]

        # Replace the old population with the new population
        population = new_population

        # Update the best individual in the population
        best_individual = max(population, key=fitness)
        best_individual = self.search_space[0] + random.uniform(-self.search_space[1], self.search_space[1])
        fitness[fitness.index(best_individual)] = self.evaluate_fitness(best_individual, func)

        # Update the population with the new individuals
        population = new_population

        return best_individual, fitness

    def select_parents(self, fitness, population, budget):
        # Select the fittest individuals to reproduce
        parents = population[:int(budget * 0.2)]
        fitness = fitness[:int(budget * 0.2)]

        # Select parents based on their fitness
        parents = sorted(population, key=lambda individual: fitness[individual], reverse=True)[:int(budget * 0.8)]
        parents = [individual for individual in parents if fitness[individual] >= self.evaluate_fitness(population[0], func)]

        return parents

    def crossover(self, parents, mutation_rate, cooling_rate):
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1[0] + 2 * (parent2[0] - parent1[0]) * random.random(), parent1[1] + 2 * (parent2[1] - parent1[1]) * random.random())
            if random.random() < mutation_rate:
                child[0] += random.uniform(-1, 1)
                child[1] += random.uniform(-1, 1)
            new_population.append(child)
        return new_population

    def evaluate_fitness(self, individual, func):
        # Evaluate the fitness of an individual
        func_value = func(individual)
        return func_value

# Example usage:
budget = 100
dim = 10
optimizer = BlackBoxOptimizer(budget, dim)

# Define a black box function
def func(x):
    return x[0]**2 + x[1]**2

# Optimize the function
best_individual, fitness = optimizer.novel_metaheuristic(func, 0.01, 0.99)

# Print the result
print(f"Best individual: {best_individual}, Fitness: {fitness}")